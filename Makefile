#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | python3 -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Installation
.PHONY: setup
setup: install pre-commit-install

.PHONY: install
install:
	poetry config virtualenvs.in-project true
	poetry env use 3.12
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	-poetry run mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: lint
lint: mypy check-codestyle dep-check

.PHONY: dep-check
dep-check:
	poetry run deptry .

.PHONY: load-data
load-data:
	mgconsole < tests/data/got.cypherl

.PHONY: test
test:
	poetry run pytest -c pyproject.toml

.PHONY: profile
profile:
	# for example:
	# make TEST=tests/module.py profile
	# or
	# make TEST="tests/module.py -k test_exists" profile
	poetry run pytest -c pyproject.toml --profile-svg $(TEST)

.PHONY: check-codestyle
check-codestyle:
	poetry run ruff check --config=./pyproject.toml .
	poetry run black --config pyproject.toml ./
	poetry run ruff check --config ./pyproject.toml --fix

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report -i 51457 -i 67599 -i 70612
	poetry run bandit -ll --recursive cudaffi tests

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

# Docs
.PHONY: docs
docs:
	poetry run mkdocs build

.PHONY: edit-docs
edit-docs:
	poetry run mkdocs serve -a 0.0.0.0:9000 -w cudaffi

# Coverage
.PHONY: coverage
coverage:
	poetry run coverage run -m pytest
	poetry run coverage lcov
	poetry run coverage report

.PHONY: coverage-server
coverage-server:
	cd htmlcov && python3 -m http.server 9099

.PHONY: doc-coverage
doc-coverage:
	poetry run interrogate -c pyproject.toml -vv cudaffi

#* Docker
# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove

# commit hooks
.PHONY: pre-commit
pre-commit: lint

.PHONY: pre-push
pre-push: doc-coverage test coverage docs
