Files under this folder are config files.
An example is `example_config_for_no_transfer.py`

Note the name of your config file must start with `config_for_` as this is the way `.gitignore` recognizes it and ignores it. For example, create a file named `config_for_hier_hurdle3_24.py`. The content of the file will be like the one inside `example_config.py`. Then run `runs/hier_hurdle3_24.py`. The file `hier_hurdle3_24.py` will import the file `config_for_hier_hurdle3_24.py` and load all the parameters in it.

Note that actual config files imported by `runs/***.py` are added in `.gitignore` so they are not uploaded to Github. Therefore changes in one person's parameters setting will not be downloaded to other collaborators' local PCs.

Always set parameters in these files instead of the `.py` files in `runs/` to avoid conflicts or, even worse, unkown changes in parameters by other people.
