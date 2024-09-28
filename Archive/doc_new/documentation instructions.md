# How to Document Models Using Sphinx

To document your models with **Sphinx**, follow these steps:

## 1. Create reStructuredText (`.rst`) files for your models

For each model or module, create an `.rst` file in your Sphinx `doc_new/source` directory.

-   Example file structure:
    ```
    docs/
    ├── source/
    │   ├── index.rst
    │   ├── TableGAN.rst
    │   ├── TableGANAdapter.rst
    │   ├── utils.rst
    ```

## 2. Structure the `.rst` Files

In each `.rst` file, document the model using the following structure:

```rst
ModelName
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Class Description
-----------------
A short description of the class and its functionality.

Methods
-------
- Method 1
  - Description: Brief explanation of what the method does.
  - Parameters:
    - `param_name`: Description of the parameter.
  - Returns:
    - Description of what the method returns.

Example
-------
Provide example usage of the class and its methods, if necessary.


## 4. Build the documentation in html.

In the doc_new root directory, run the command 'make html' - which should generate the documentation in a folder called _build. The items can be accessed by pressing on opening one of the html files under the directory called html under _build.
```
