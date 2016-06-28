set -e # Exit on failure

echo Tests
nosetests --cover-html --cover-inclusive --cover-erase --cover-tests --cover-package=vision vision