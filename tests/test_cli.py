from pathlib import Path

from mcwm.cli import _recording_group, _split_files


def test_recording_split_keeps_player_sessions_disjoint() -> None:
    files = [
        Path(f"cheeky-cornflower-setter-{player}-20220421-09263{take}.jsonl")
        for player in ("aaaa", "bbbb", "cccc", "dddd")
        for take in range(2)
    ]
    train, validation, test = _split_files(files, seed=7)
    split_groups = [
        {_recording_group(path) for path in split}
        for split in (train, validation, test)
    ]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert sum(map(len, (train, validation, test))) == len(files)
