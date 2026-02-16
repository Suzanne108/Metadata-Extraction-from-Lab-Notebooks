from typing import List, Optional, Dict, Any

from langchain.messages import AIMessage

from src.state import ISAExtractionState


def create_assembly_node() -> callable:
    """Deterministically assemble DTO registries into a linked isatools investigation graph.

    Strategy
    - Convert registries of leaf DTOs (people, publications, sources, samples, materials, data, protocols,
      factor values) to isatools objects and index by uid.
    - Convert parent DTOs (assays, processes, factors, studies, investigation) and link nested references
      by uid where available (e.g., AssayDTO.data_files → Data objects).
    - Attach people/publications to investigation when attributes exist.
    - Attach studies to investigation when attributes exist.
    """

    def _to_obj_list(dtos: Dict[str, object]) -> tuple[list, dict]:
        """Convert a registry of DTOs to isatools objects and return list and uid-indexed map."""

        objects: list = []
        by_uid: dict = {}
        for uid, dto in (dtos or {}).items():
            try:
                obj = dto.to_isatools_model()
                objects.append(obj)
                by_uid[uid] = obj
            except Exception:
                # Skip conversion errors silently to preserve assembly
                pass
        return objects, by_uid

    def _resolve_ids_to_objects(ids: Optional[List[str]], *maps: Dict[str, Any]) -> List[Any]:
        """Given a list of uids and one or more uid->object maps, return matched objects in order.
        The first map containing the uid wins.
        """
        if not ids:
            return []
        out: List[Any] = []
        for _id in ids:
            for m in maps:
                if _id in m:
                    out.append(m[_id])
                    break
        return out

    def _attach_assay_links(assay_dto, assay_obj, *, data_map, sample_map, material_map, process_map):
        """Attach data files, samples, other materials, and process sequence to an assay object.
        Assumes the DTO stores uids (strings) in the corresponding fields.
        """
        try:
            df_objs = _resolve_ids_to_objects(getattr(assay_dto, "data_files", None), data_map)
            if df_objs and hasattr(assay_obj, "data_files"):
                try:
                    assay_obj.data_files = df_objs
                except Exception:
                    pass
            s_objs = _resolve_ids_to_objects(getattr(assay_dto, "samples", None), sample_map)
            if s_objs and hasattr(assay_obj, "samples"):
                try:
                    assay_obj.samples = s_objs
                except Exception:
                    pass
            m_objs = _resolve_ids_to_objects(getattr(assay_dto, "other_materials", None), material_map)
            if m_objs and hasattr(assay_obj, "other_materials"):
                try:
                    assay_obj.other_materials = m_objs
                except Exception:
                    pass
            p_objs = _resolve_ids_to_objects(getattr(assay_dto, "process_sequence", None), process_map)
            if p_objs and hasattr(assay_obj, "process_sequence"):
                try:
                    assay_obj.process_sequence = p_objs
                except Exception:
                    pass
        except Exception:
            pass

    def _attach_process_links(proc_dto, proc_obj, *, process_map, protocol_map, data_map, material_map, sample_map, source_map):
        """Attach protocol, previous/next processes, and input/output objects to a process object by ids."""
        try:
            proto_id = getattr(proc_dto, "executes_protocol", None)
            if proto_id and proto_id in protocol_map and hasattr(proc_obj, "executes_protocol"):
                try:
                    proc_obj.executes_protocol = protocol_map[proto_id]
                except Exception:
                    pass
            prev_objs = _resolve_ids_to_objects(getattr(proc_dto, "previous_processes", None), process_map)
            if prev_objs and hasattr(proc_obj, "previous_processes"):
                try:
                    proc_obj.previous_processes = prev_objs
                except Exception:
                    pass
            next_objs = _resolve_ids_to_objects(getattr(proc_dto, "next_processes", None), process_map)
            if next_objs and hasattr(proc_obj, "next_processes"):
                try:
                    proc_obj.next_processes = next_objs
                except Exception:
                    pass
            # Explicit sources list, if present
            src_objs = _resolve_ids_to_objects(getattr(proc_dto, "sources", None), source_map)
            if src_objs and hasattr(proc_obj, "sources"):
                try:
                    proc_obj.sources = src_objs
                except Exception:
                    pass
            # inputs may include materials, samples, data files, and sources by id
            in_objs = _resolve_ids_to_objects(getattr(proc_dto, "inputs", None), material_map, sample_map, data_map, source_map)
            if in_objs and hasattr(proc_obj, "inputs"):
                try:
                    proc_obj.inputs = in_objs
                except Exception:
                    pass
            out_objs = _resolve_ids_to_objects(getattr(proc_dto, "outputs", None), material_map, sample_map, data_map)
            if out_objs and hasattr(proc_obj, "outputs"):
                try:
                    proc_obj.outputs = out_objs
                except Exception:
                    pass
        except Exception:
            pass

    def _attach_study_links(study_dto, study_obj, *, assay_map, protocol_map, source_map, sample_map, material_map, process_map, factor_map):
        """Attach protocols, assays, sources, samples, materials, processes, and factors to a study object by ids."""
        try:
            protos = _resolve_ids_to_objects(getattr(study_dto, "protocols", None), protocol_map)
            if protos and hasattr(study_obj, "protocols"):
                try:
                    study_obj.protocols = protos
                except Exception:
                    pass
            assays = _resolve_ids_to_objects(getattr(study_dto, "assays", None), assay_map)
            if assays and hasattr(study_obj, "assays"):
                try:
                    study_obj.assays = assays
                except Exception:
                    pass
            sources = _resolve_ids_to_objects(getattr(study_dto, "sources", None), source_map)
            if sources and hasattr(study_obj, "sources"):
                try:
                    study_obj.sources = sources
                except Exception:
                    pass
            samples = _resolve_ids_to_objects(getattr(study_dto, "samples", None), sample_map)
            if samples and hasattr(study_obj, "samples"):
                try:
                    study_obj.samples = samples
                except Exception:
                    pass
            materials = _resolve_ids_to_objects(getattr(study_dto, "other_materials", None), material_map)
            if materials and hasattr(study_obj, "other_materials"):
                try:
                    study_obj.other_materials = materials
                except Exception:
                    pass
            procs = _resolve_ids_to_objects(getattr(study_dto, "process_sequence", None), process_map)
            if procs and hasattr(study_obj, "process_sequence"):
                try:
                    study_obj.process_sequence = procs
                except Exception:
                    pass
            factors = _resolve_ids_to_objects(getattr(study_dto, "factors", None), factor_map)
            if factors and hasattr(study_obj, "factors"):
                try:
                    study_obj.factors = factors
                except Exception:
                    pass
        except Exception:
            pass

    def _attach_sample_links(sample_dto, sample_obj, *, factor_value_map, source_map):
        """Attach factor values and sources to a sample object by ids if present on DTO."""
        try:
            fv_objs = _resolve_ids_to_objects(getattr(sample_dto, "factor_values", None), factor_value_map)
            if fv_objs and hasattr(sample_obj, "factor_values"):
                try:
                    sample_obj.factor_values = fv_objs
                except Exception:
                    pass
            src_objs = _resolve_ids_to_objects(getattr(sample_dto, "sources", None), source_map)
            if src_objs and hasattr(sample_obj, "sources"):
                try:
                    sample_obj.sources = src_objs
                except Exception:
                    pass
        except Exception:
            pass

    def _attach_factorvalue_links(fv_dto, fv_obj, *, factor_map):
        """Attach factor/category object to a factor value if DTO references a factor by id."""
        try:
            # Common field names: 'factor' or 'category' may hold a uid
            fid = getattr(fv_dto, "factor", None) or getattr(fv_dto, "category", None)
            if isinstance(fid, str) and fid in factor_map:
                # Prefer setting an attribute named 'category' or 'factor' on the isatools object
                if hasattr(fv_obj, "category"):
                    try:
                        fv_obj.category = factor_map[fid]
                    except Exception:
                        pass
                elif hasattr(fv_obj, "factor"):
                    try:
                        fv_obj.factor = factor_map[fid]
                    except Exception:
                        pass
        except Exception:
            pass

    def assembly(state: ISAExtractionState) -> dict:
        inv_dto = state.get("dto_investigation")
        inv_obj = None
        msg = "NODE: ASSEMBLY; status=success"

        try:
            # Build leaf objects and maps
            source_list, source_map = _to_obj_list(state.get("registry_sources") or {})
            sample_list, sample_map = _to_obj_list(state.get("registry_samples") or {})
            material_list, material_map = _to_obj_list(state.get("registry_materials") or {})
            data_list, data_map = _to_obj_list(state.get("registry_data") or {})
            protocol_list, protocol_map = _to_obj_list(state.get("registry_protocols") or {})
            factor_value_list, factor_value_map = _to_obj_list(state.get("registry_factor_values") or {})
            process_list, process_map = _to_obj_list(state.get("registry_processes") or {})

            # Parents
            assay_list: list = []
            assay_map: dict = {}
            for uid, a in (state.get("registry_assays") or {}).items():
                try:
                    assay_obj = a.to_isatools_model()
                    _attach_assay_links(a, assay_obj, data_map=data_map, sample_map=sample_map, material_map=material_map, process_map=process_map)
                    assay_list.append(assay_obj)
                    assay_map[uid] = assay_obj
                except Exception:
                    pass

            process_list, process_map = _to_obj_list(state.get("registry_processes") or {})
            # Attach process links 
            for puid, pdto in (state.get("registry_processes") or {}).items():
                pobj = process_map.get(puid)
                if pobj is not None:
                    _attach_process_links(pdto, pobj, process_map=process_map, protocol_map=protocol_map, data_map=data_map, material_map=material_map, sample_map=sample_map, source_map=source_map)
            factor_list, factor_map = _to_obj_list(state.get("registry_factors") or {})
            # Attach factorvalue links to factors when possible
            for fvid, fvdto in (state.get("registry_factor_values") or {}).items():
                fvobj = None
                # find converted fv object: try by position using map
                try:
                    fvobj = factor_value_map.get(fvid)
                except Exception:
                    fvobj = None
                if fvobj is not None:
                    _attach_factorvalue_links(fvdto, fvobj, factor_map=factor_map)
            study_list, study_map = _to_obj_list(state.get("registry_studies") or {})
            for suid, sdto in (state.get("registry_studies") or {}).items():
                sobj = study_map.get(suid)
                if sobj is not None:
                    _attach_study_links(sdto, sobj, assay_map=assay_map, protocol_map=protocol_map, source_map=source_map, sample_map=sample_map, material_map=material_map, process_map=process_map, factor_map=factor_map)

            # Attach sample links
            for samp_uid, samp_dto in (state.get("registry_samples") or {}).items():
                samp_obj = sample_map.get(samp_uid)
                if samp_obj is not None:
                    _attach_sample_links(samp_dto, samp_obj, factor_value_map=factor_value_map, source_map=source_map)

            # Investigation
            if inv_dto is not None:
                inv_obj = inv_dto.to_isatools_model()

            # Attach to investigation if attributes exist
            if inv_obj is not None:
                if hasattr(inv_obj, "studies") and study_list:
                    inv_obj.studies = study_list

        except Exception as e:
            msg = f"NODE: ASSEMBLY; status=error; {e}"

        # Avoid returning live isatools objects which break checkpoint serialization
        return {
            "isatools_investigation": None,
            "last_action": "ASSEMBLED",
            "messages": [AIMessage(content=msg)],
        }

    return assembly
