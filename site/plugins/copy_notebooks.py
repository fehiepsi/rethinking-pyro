import glob
import io
import os

import nbformat

from nikola.plugin_categories import Task
from nikola import utils


class CopyNotebooks(Task):
    """Copy notebooks into pages."""

    name = "copy_notebooks"

    def set_site(self, site):
        self.site = site
        self.inject_dependency("render_posts", "copy_notebooks")
        super(CopyNotebooks, self).set_site(site)

    def gen_tasks(self):
        kw = {
            "filters": self.site.config["FILTERS"],
            "notebooks_folder": self.site.config["NOTEBOOKS_FOLDER"],
        }

        nb_files = sorted(glob.glob(os.path.join(kw["notebooks_folder"], "*.ipynb")))
        for i, src_file in enumerate(nb_files):
            dst_file = os.path.join("pages", os.path.basename(src_file).replace("_", "-"))
            prev_nb, next_nb = None, None
            if i > 0:
                prev_nb = nb_files[i - 1]
            if i < len(nb_files) - 1:
                next_nb = nb_files[i + 1]
            task = {
                "basename": self.name,
                "name": dst_file,
                "file_dep": [src_file],
                "targets": [dst_file],
                "actions": [(copynb, (src_file, dst_file, prev_nb, next_nb))],
                "uptodate": [utils.config_changed(kw, "copy_notebooks")],
                "clean": True,
            }
            yield utils.apply_filters(task, kw["filters"])


def copynb(src_file, dst_file, prev_nb, next_nb):
    # navigation
    prev_link = get_nb_link(prev_nb) if prev_nb is not None else ""
    next_link = get_nb_link(next_nb) if next_nb is not None else ""
    prev_title = get_nb_title(prev_nb) if prev_nb is not None else ""
    next_title = get_nb_title(next_nb) if next_nb is not None else ""

    # read notebook
    with io.open(src_file, "r", encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    # add title
    title = get_nb_link(src_file)
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("#"):
            title = cell.source.splitlines()[0].lstrip("#").strip()
            cell.source = "\n".join(cell.source.splitlines()[1:])
            break
    nb.metadata.nikola = {"title": title}

    # add navigation
    nav_template = "<!-- NAVIGATION -->\n< [{}]({}) | [Contents](index.html) | [{}]({}) >"
    navigation = nav_template.format(prev_title, prev_link, next_title, next_link)
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(navigation, metadata={"navigation": True}))
    nb.cells.append(nbformat.v4.new_markdown_cell(navigation, metadata={"navigation": True}))

    # write notebook
    with io.open(dst_file, "w", encoding="utf8") as f:
        nbformat.write(nb, f)


def get_nb_link(nbfile):
    return os.path.basename(nbfile).replace("_", "-").replace(".ipynb", ".html")


def get_nb_title(nbfile):
    with io.open(nbfile, "r", encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("#"):
            return cell.source.lstrip("#").splitlines()[0].strip()

    return os.path.basename(nbfile)
