smoke-test:
	echo "smoke testing track feature"
	uv run choreopath track examples/fall-recovery-4.mp4 test-data/test.csv
	echo "smoke testing draw feature"
	uv run choreopath draw test-data/test.csv test-data/test.svg
	echo "smoke testing analyze feature"
	uv run choreopath analyze test-data/test.csv
	echo "smoke testing overlay feature"
	uv run choreopath overlay examples/fall-recovery-4.mp4 test-data/overlay.mp4
	echo "smoke testing overlay feature"
	uv run choreopath overlay --paths-only examples/fall-recovery-4.mp4 test-data/overlay-paths-only.mp4
