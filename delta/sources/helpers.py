# -*- Encoding: UTF-8 -*-

from sources.loader_kstarecei import loader_kstarecei


def get_loader(cfg_all):
    """Returns data loader appropriate for the configured diagnostic.

    Args:
        cfg_all (dict):
            Configuration dictionary

    Returns:
        dataloader (dataloader):
            Dataloader object
    """
    if cfg_all["diagnostic"]["name"] == "kstarecei":
        return loader_kstarecei(cfg_all)
    else:
        raise ValueError("No dataloader for " + cfg_all["diagnostic"]["name"])


# End of file sources/helpers.py
