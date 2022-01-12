# parmscan

Provide tools to plot parameter scan results that are wrapped up in a high-dimensional numpy array

* repository: https://github.com/benmaier/parmscan/
* documentation: http://parmscan.benmaier.org/

```python
from parmscan.sqrt import get_sqrt_2

print(get_sqrt_2())
```

## Install

    pip install parmscan

`parmscan` was developed and tested for 

* Python 3.6
* Python 3.7
* Python 3.8

So far, the package's functionality was tested on Mac OS X and CentOS only.

## Dependencies

`parmscan` directly depends on the following packages which will be installed by `pip` during the installation process

* `numpy>=1.17`

## Documentation

The full documentation is available at [parmscan.benmaier.org](http://parmscan.benmaier.org).

## Changelog

Changes are logged in a [separate file](https://github.com/benmaier/parmscan/blob/main/CHANGELOG.md).

## License

This project is licensed under the [MIT License](https://github.com/benmaier/parmscan/blob/main/LICENSE).
Note that this excludes any images/pictures/figures shown here or in the documentation.

## Contributing

If you want to contribute to this project, please make sure to read the [code of conduct](https://github.com/benmaier/parmscan/blob/main/CODE_OF_CONDUCT.md) and the [contributing guidelines](https://github.com/benmaier/parmscan/blob/main/CONTRIBUTING.md). In case you're wondering about what to contribute, we're always collecting ideas of what we want to implement next in the [outlook notes](https://github.com/benmaier/parmscan/blob/main/OUTLOOK.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)

## Dev notes

Fork this repository, clone it, and install it in dev mode.

```bash
git clone git@github.com:YOURUSERNAME/parmscan.git
make
```

If you want to upload to PyPI, first convert the new `README.md` to `README.rst`

```bash
make readme
```

It will give you warnings about bad `.rst`-syntax. Fix those errors in `README.rst`. Then wrap the whole thing 

```bash
make pypi
```

It will probably give you more warnings about `.rst`-syntax. Fix those until the warnings disappear. Then do

```bash
make upload
```
