from pathlib import Path
import json
from click.testing import CliRunner

from src.cli import cli


def test_cli_list_and_show(tmp_path: Path):
    # Build canonical structure
    base = tmp_path / "outputs"
    proj = base / "projects" / "Demo"
    asset_id = "chair_L2_000001"
    asset_dir = proj / "assets" / "session_1" / "raw" / asset_id
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset_metadata.json").write_text(json.dumps({"asset_id": asset_id, "asset_name": "chair"}))

    # Build index
    index = base / "asset_index.json"
    index.write_text(json.dumps({asset_id: str(asset_dir)}))

    runner = CliRunner()
    with runner.isolated_filesystem(tmp_dir=str(tmp_path)):
        # list-assets
        res = runner.invoke(cli, ["list-assets", "--format", "json"])
        assert res.exit_code == 0, res.output
        data = json.loads(res.output)
        ids = [a["asset_id"] for a in data["assets"]]
        assert asset_id in ids

        # show
        res2 = runner.invoke(cli, ["show", asset_id])
        assert res2.exit_code == 0, res2.output
        assert "Asset Directory" in res2.output
