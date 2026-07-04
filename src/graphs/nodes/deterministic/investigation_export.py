from typing import List, Dict, Any, Optional

from pathlib import Path
import json
import logging
import traceback

from langchain.messages import AIMessage

from isatools.isajson import ISAJSONEncoder
from isatools.model import Investigation

from src.graphs.nodes.base_node import DeterministicGraphNode
from src.graphs.types.Node import GraphComponentId
from src.state import ISAExtractionState


LOGGER = logging.getLogger("isa.investigation_export")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)

class InvestigationExportNode(DeterministicGraphNode):
    """Collapse all created objects into a single isatools Investigation and dump to ISA-JSON.

    This node favors simplicity over perfect linking:
    - Converts all DTO registries to isatools.model objects.
    - Attaches all studies found to the Investigation.
    - For each Study, attaches all assays, protocols, sources, samples, materials, processes, and factors present.
    - Optionally sets basic nested fields on assays if attributes exist.
    - Writes the Investigation to `output/<doc_stem>_extracted_isa.json` via ISAJSONEncoder.
    """

    def __init__(self) -> None:
        super().__init__(node_id=GraphComponentId.EXPORT.value)

    def _log_exception(self, context: str, exc: Optional[BaseException] = None):
        # Always print a clear, actionable error with traceback to the console
        if exc is None:
            exc = Exception("Unknown error")
        LOGGER.error("Error during: %s: %s", context, exc)
        tb = traceback.format_exc()
        # Ensure traceback is printed even if logging handler filters it
        print(f"TRACEBACK [{context}]:\n{tb}")

    def _to_obj_list(self, dtos: Dict[str, object]) -> List[Any]:
        objects: List[Any] = []
        for key, dto in (dtos or {}).items():
            try:
                obj = dto.to_isatools_model()
                objects.append(obj)
            except Exception as e:
                self._log_exception(f"DTO to_isatools_model conversion (key={key}, type={type(dto).__name__})", e)
                # Skip conversion errors to keep export robust
                continue
        return objects

    def _safe_set(self, obj: Any, attr: str, val: Any):
        try:
            if hasattr(obj, attr):
                setattr(obj, attr, val)
        except Exception as e:
            self._log_exception(f"setattr failed on {type(obj).__name__}.{attr}", e)

    def _document_stem(self, state: ISAExtractionState) -> str:
        try:
            doc = state.get("document")
            # PyPDFLoader returns a list of Document objects; prefer first
            first = (doc[0] if isinstance(doc, list) and doc else doc)
            meta = getattr(first, "metadata", {}) or {}
            src = meta.get("source") or meta.get("path")
            if src:
                return Path(src).stem
        except Exception as e:
            self._log_exception("deriving document stem from state.document.metadata", e)
        return "isa_export"

    def run(self, state: ISAExtractionState) -> dict:
        msg = "NODE: INVESTIGATION_EXPORT; status=success"
        inv_obj: Optional[Investigation] = None

        try:
            # Convert registries to lists of isatools objects
            study_list = self._to_obj_list(state.get("registry_studies") or {})
            assay_list = self._to_obj_list(state.get("registry_assays") or {})
            protocol_list = self._to_obj_list(state.get("registry_protocols") or {})
            source_list = self._to_obj_list(state.get("registry_sources") or {})
            sample_list = self._to_obj_list(state.get("registry_samples") or {})
            material_list = self._to_obj_list(state.get("registry_materials") or {})
            process_list = self._to_obj_list(state.get("registry_processes") or {})
            factor_list = self._to_obj_list(state.get("registry_factors") or {})
            data_list = self._to_obj_list(state.get("registry_data") or {})

            # Build or convert investigation
            inv_dto = state.get("dto_investigation")
            try:
                inv_obj = inv_dto.to_isatools_model() if inv_dto is not None else Investigation()
            except Exception as e:
                self._log_exception("InvestigationDTO to_isatools_model conversion", e)
                inv_obj = Investigation()

            # Attach all studies
            if study_list:
                self._safe_set(inv_obj, "studies", study_list)

            # Attach discovered objects only to the FIRST study
            first_study = study_list[0] if study_list else None
            if first_study:
                self._safe_set(first_study, "assays", assay_list)
                self._safe_set(first_study, "protocols", protocol_list)
                self._safe_set(first_study, "sources", source_list)
                self._safe_set(first_study, "samples", sample_list)
                self._safe_set(first_study, "other_material", material_list)
                self._safe_set(first_study, "process_sequence", process_list)
                self._safe_set(first_study, "factors", factor_list)

                # ------------------------------------------------------------------
                # Collect characteristic and unit categories
                #
                # Characteristics on sources/samples/materials are currently created
                # via MaterialAttributeValueDTO, which sets:
                #   - category -> an ontology-like object
                #   - unit     -> an ontology-like object (optional)
                #
                # ISAJSONEncoder expects these category/unit "vocabularies" to be
                # attached to the study (characteristic_categories, unit_categories).
                # If they are missing, the exported ISA-JSON only contains bare
                # "@id" references without any annotationValue. Here we collect all
                # unique category and unit objects from the attached materials and
                # register them on the first study so they are fully serialised.
                # ------------------------------------------------------------------
                try:
                    characteristic_categories = []
                    unit_categories = []

                    def _add_unique(target_list: list, obj: Any):
                        if obj is None:
                            return
                        # Identity-based de-duplication is sufficient here
                        if obj not in target_list:
                            target_list.append(obj)

                    def _collect_from_objects(objs: list[object]):
                        for o in objs or []:
                            chars = getattr(o, "characteristics", None) or []
                            for ch in chars:
                                cat = getattr(ch, "category", None)
                                _add_unique(characteristic_categories, cat)

                                unit = getattr(ch, "unit", None)
                                _add_unique(unit_categories, unit)

                    _collect_from_objects(source_list)
                    _collect_from_objects(sample_list)
                    _collect_from_objects(material_list)

                    if characteristic_categories and hasattr(first_study, "characteristic_categories"):
                        self._safe_set(first_study, "characteristic_categories", characteristic_categories)

                    if unit_categories and hasattr(first_study, "unit_categories"):
                        self._safe_set(first_study, "unit_categories", unit_categories)

                except Exception as e:
                    self._log_exception("collecting characteristic/unit categories for study", e)

            # Set data files only on the FIRST assay
            first_assay = assay_list[0] if assay_list else None
            if first_assay:
                self._safe_set(first_assay, "data_files", data_list)

            # Determine output path based on input document name
            stem = self._document_stem(state)
            out_dir = Path.cwd() / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_extracted_isa.json"

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(inv_obj, f, cls=ISAJSONEncoder, sort_keys=True, indent=4, separators=(",", ": "))
                msg += f"; wrote={out_path}"
            except Exception as e:
                self._log_exception(f"writing ISA-JSON to {out_path}", e)
                msg = f"NODE: INVESTIGATION_EXPORT; status=error; write_failed={e} (see console)"

        except Exception as e:
            self._log_exception("top-level export() execution", e)
            msg = f"NODE: INVESTIGATION_EXPORT; status=error; {e} (see console)"

        return {
            "last_action": "EXPORTED",
            "messages": [AIMessage(content=msg)],
        }
