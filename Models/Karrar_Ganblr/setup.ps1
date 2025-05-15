echo "rm -rf old dist"
rm -Recurse -Force dist
echo "run: python setup.py sdist bdist_wheel"
python setup.py sdist bdist_wheel
echo "run: upload!"
twine upload dist/*