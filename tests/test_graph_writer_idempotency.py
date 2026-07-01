from __future__ import annotations

from app.domain.ontology import Evidence, Measurement
from app.graph.graph_models import ExperimentFact
from app.graph.graph_writer import GraphWriteStats, GraphWriter, deterministic_measurement_id


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return []


class BackfillSession(FakeSession):
    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "RETURN meas.measurement_id AS measurement_id" in query:
            return [
                {
                    "measurement_id": "legacy-m1",
                    "value": 77.0,
                    "raw_value": "77",
                    "unit": "ksi",
                    "property": "прочность",
                }
            ]
        return []


class FakeGraphDB:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def session(self):
        session = self._session

        class Context:
            def __enter__(self):
                return session

            def __exit__(self, exc_type, exc, tb):
                return False

        return Context()


def _fact() -> ExperimentFact:
    evidence = Evidence(document_id="doc-1", chunk_id="chunk-1", source_name="source.txt", page=1, quote="quote")
    return ExperimentFact(
        experiment_id="EXP-VT6-AN",
        materials=["ВТ6"],
        regimes=["отжиг"],
        measurements=[
            Measurement(property_name="прочность", value=1120.0, raw_value="1120", unit="MPa", effect="increase", confidence=0.9, evidence=[evidence])
        ],
        equipment=["Вакуумная печь"],
        laboratories=["Лаборатория легких сплавов"],
        conclusions=["отжиг повысил прочность"],
        evidence=[evidence],
        source_chunk_ids=["chunk-1"],
    )


def test_measurement_id_is_deterministic() -> None:
    first = deterministic_measurement_id("EXP", "ВТ6", "отжиг", "прочность", 1120.0, "MPa", "chunk-1")
    second = deterministic_measurement_id("EXP", "ВТ6", "отжиг", "прочность", 1120.0, "MPa", "chunk-1")
    assert first == second
    assert first.startswith("measurement_")


def test_writer_uses_merge_and_stable_ids() -> None:
    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_experiment(session, _fact(), stats)
    writer.write_experiment(session, _fact(), stats)

    query_text = "\n".join(query for query, _ in session.calls)
    assert "MERGE (e:Experiment" in query_text
    assert "MERGE (m:Material" in query_text
    assert "MERGE (meas:Measurement" in query_text
    assert "CREATE " not in query_text
    measurement_ids = [params["measurement_id"] for _, params in session.calls if "measurement_id" in params]
    assert len(set(measurement_ids)) == 1


def test_writer_persists_normalized_measurement_fields() -> None:
    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_experiment(session, _fact(), stats)

    measurement_params = next(params for _, params in session.calls if "measurement_id" in params and "value_normalized" in params)
    assert measurement_params["value_original"] == 1120.0
    assert measurement_params["unit_original"] == "MPa"
    assert measurement_params["value_normalized"] == 1120.0
    assert measurement_params["unit_normalized"] == "MPa"
    assert measurement_params["normalization_family"] == "strength"


def test_writer_backfills_legacy_normalized_measurement_fields() -> None:
    session = BackfillSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.backfill_normalized_measurements(session, stats)

    update_params = next(params for query, params in session.calls if "MATCH (meas:Measurement {measurement_id: $measurement_id})" in query)
    assert update_params["measurement_id"] == "legacy-m1"
    assert update_params["value_original"] == 77.0
    assert update_params["unit_original"] == "ksi"
    assert abs(update_params["value_normalized"] - 530.896289) < 0.001
    assert update_params["unit_normalized"] == "MPa"
    assert update_params["normalization_family"] == "strength"
    assert stats.normalized_measurements_backfilled == 1
