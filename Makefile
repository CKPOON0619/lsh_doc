setup: requirements.txt
	pip install -r requirements.txt

test:
	python -m unittest

demo:
	python -m demo