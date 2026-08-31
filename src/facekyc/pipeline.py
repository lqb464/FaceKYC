"""One raw-image-to-decision pipeline shared by API and local evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from facekyc.artifacts import ArtifactError, load_bundle
from facekyc.config import Settings, load_settings
from facekyc.contracts import ImageQualityReport, VerificationResult
from facekyc.models import CNNPAD, FaceNetRecognizer, MTCNNFaceDetector
from facekyc.quality import assess_image_quality


class FaceKYCPipeline:
    """Orchestrate quality, face detection, PAD, and 1:1 verification."""

    def __init__(
        self,
        *,
        settings: Settings,
        bundle: dict[str, Any],
        detector: Any,
        recognizer: Any,
        liveness: Any,
    ):
        self.settings = settings
        self.bundle = bundle
        self.detector = detector
        self.recognizer = recognizer
        self.liveness = liveness
        self.verification_threshold = float(bundle["verification"]["threshold"])
        self.liveness_threshold = float(bundle["liveness"]["threshold"])
        self.model_version = str(bundle["model_version"])
        self.deployment_status = str(bundle["deployment_status"])

    @classmethod
    def from_artifacts(
        cls,
        config_path: str | Path | None = None,
        bundle_path: str | Path | None = None,
        *,
        device: str | None = None,
        require_approved: bool = True,
    ) -> FaceKYCPipeline:
        settings = load_settings(config_path)
        resolved_bundle = settings.resolve(bundle_path or settings.artifact.bundle_path)
        bundle = load_bundle(resolved_bundle, verify_weights=True)
        if require_approved and bundle["deployment_status"] != "approved":
            raise ArtifactError(
                f"Model bundle is not approved for serving: {bundle['deployment_status']}"
            )
        detector = MTCNNFaceDetector(settings.detection, device=device)
        recognizer = FaceNetRecognizer(bundle["verification"]["pretrained_dataset"], device=device)
        liveness = CNNPAD(
            bundle["_resolved_liveness_weights_path"],
            architecture=bundle["liveness"]["architecture"],
            live_class_index=bundle["liveness"]["live_class_index"],
            input_size=bundle["liveness"]["input_size"],
            threshold=bundle["liveness"]["threshold"],
            device=device,
        )
        return cls(
            settings=settings,
            bundle=bundle,
            detector=detector,
            recognizer=recognizer,
            liveness=liveness,
        )

    def _result(
        self,
        *,
        decision: str,
        reasons: list[str],
        quality: dict[str, ImageQualityReport],
        warnings: list[str],
        similarity: float | None = None,
        liveness: float | None = None,
    ) -> dict[str, Any]:
        return VerificationResult(
            status="completed",
            decision=decision,
            reason_codes=tuple(dict.fromkeys(reasons)),
            similarity_score=None if similarity is None else round(similarity, 6),
            liveness_score=None if liveness is None else round(liveness, 6),
            thresholds={
                "similarity": self.verification_threshold,
                "liveness": self.liveness_threshold,
            },
            image_quality={name: report.to_dict() for name, report in quality.items()},
            input_warnings=tuple(dict.fromkeys(warnings)),
            model_version=self.model_version,
            deployment_status=self.deployment_status,
        ).to_dict()

    def verify(self, id_image: Image.Image, selfie_image: Image.Image) -> dict[str, Any]:
        quality = {
            "id_image": assess_image_quality(id_image, self.settings.input),
            "selfie_image": assess_image_quality(selfie_image, self.settings.input),
        }
        quality_reasons = [
            f"{name}_{reason}"
            for name, report in quality.items()
            for reason in report.rejection_reasons
        ]
        warnings = [
            f"{name}_{warning}" for name, report in quality.items() for warning in report.warnings
        ]
        if quality_reasons:
            return self._result(
                decision="recapture",
                reasons=quality_reasons,
                quality=quality,
                warnings=warnings,
            )

        id_face = self.detector.detect(id_image)
        selfie_face = self.detector.detect(selfie_image)
        if id_face is None or selfie_face is None:
            reasons = []
            if id_face is None:
                reasons.append("id_image_face_not_detected")
            if selfie_face is None:
                reasons.append("selfie_image_face_not_detected")
            return self._result(
                decision="recapture", reasons=reasons, quality=quality, warnings=warnings
            )

        for name, observation in (("id_image", id_face), ("selfie_image", selfie_face)):
            warnings.extend(f"{name}_{warning}" for warning in observation.warnings)
        if self.settings.detection.reject_multiple_faces and (
            id_face.face_count > 1 or selfie_face.face_count > 1
        ):
            return self._result(
                decision="recapture",
                reasons=["multiple_faces_detected"],
                quality=quality,
                warnings=warnings,
            )
        low_confidence = []
        if id_face.probability < self.settings.detection.min_probability:
            low_confidence.append("id_image_low_detection_confidence")
        if selfie_face.probability < self.settings.detection.min_probability:
            low_confidence.append("selfie_image_low_detection_confidence")
        if low_confidence:
            return self._result(
                decision="recapture", reasons=low_confidence, quality=quality, warnings=warnings
            )

        liveness_score = float(self.liveness.score(selfie_face.crop))
        first_embedding = self.recognizer.embedding(id_face.tensor)
        second_embedding = self.recognizer.embedding(selfie_face.tensor)
        similarity_score = float(self.recognizer.similarity(first_embedding, second_embedding))

        if (
            self.settings.decision.review_on_uncertain_score
            and abs(liveness_score - self.liveness_threshold)
            <= self.settings.liveness.uncertainty_margin
        ):
            return self._result(
                decision="manual_review",
                reasons=["uncertain_liveness_score"],
                quality=quality,
                warnings=warnings,
                similarity=similarity_score,
                liveness=liveness_score,
            )
        if liveness_score < self.liveness_threshold:
            return self._result(
                decision=self.settings.decision.failed_pad_action,
                reasons=["presentation_attack_suspected"],
                quality=quality,
                warnings=warnings,
                similarity=similarity_score,
                liveness=liveness_score,
            )
        if (
            self.settings.decision.review_on_uncertain_score
            and abs(similarity_score - self.verification_threshold)
            <= self.settings.verification.uncertainty_margin
        ):
            return self._result(
                decision="manual_review",
                reasons=["uncertain_similarity_score"],
                quality=quality,
                warnings=warnings,
                similarity=similarity_score,
                liveness=liveness_score,
            )
        if similarity_score < self.verification_threshold:
            return self._result(
                decision=self.settings.decision.low_similarity_action,
                reasons=["face_pair_below_threshold"],
                quality=quality,
                warnings=warnings,
                similarity=similarity_score,
                liveness=liveness_score,
            )
        if warnings and self.settings.decision.review_on_input_warning:
            return self._result(
                decision="manual_review",
                reasons=["input_quality_warning"],
                quality=quality,
                warnings=warnings,
                similarity=similarity_score,
                liveness=liveness_score,
            )
        return self._result(
            decision="verified",
            reasons=["all_checks_passed"],
            quality=quality,
            warnings=warnings,
            similarity=similarity_score,
            liveness=liveness_score,
        )
