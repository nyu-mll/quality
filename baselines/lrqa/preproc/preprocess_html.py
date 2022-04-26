import argparse
import bs4
import logging
import os

from bs4 import BeautifulSoup

PERMITTED_TAGS = [
    "html",
    "[document]",
    "p",
    "b",
    "i",
    "u",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ol",
    "ul",
    "pre",
    "br",
]

TAGS_TO_DECOMPOSE = [
    "img",
    "table",
    "head",
]


def get_initial_alt_text(node):
    if node.name != "img":
        return None
    ## Check that the alt text contains only a single letter
    if "alt" in node.attrs and len(node["alt"]) == 1 and node["alt"].isalpha():
        return node
    return None


def has_text(input_str):
    return input_str is not None and len(input_str.strip())


def get_text_nodes(input_node):
    if isinstance(input_node, bs4.element.NavigableString):
        return [input_node] if has_text(input_node.string) else []
    return input_node.find_all(string=has_text)


def add_letter_from_initial_img(node):
    """If the text has an initial/drop cap with alt text, then extract the letter from
    the initial and pre-pend it to the next node containing text."""
    found_letter_img = False
    letter_img = get_initial_alt_text(node)
    if letter_img is None:
        return False
    # Find the next node in the doctree to have text in it to prepend the initial to.
    node_to_change = None
    # Check current node.
    if len(node.find_all(string=has_text)):
        node_to_change = node.find_all(string=has_text)[0]
    # Check sibling nodes and subtrees.
    elif True in [len(get_text_nodes(s)) > 0 for s in node.next_siblings]:
        for s in node.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break
    # Check parent node.
    elif node.parent is not None and has_text(node.parent.string):
        node_to_change = node.parent
    # Check siblings of parent node and subtrees.
    elif node.parent is not None and True in [len(get_text_nodes(s)) > 0 for s in node.parent.next_siblings]:
        for s in node.parent.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break
    # Check grandparent node.
    elif node.parent.parent is not None and has_text(node.parent.parent.string):
        node_to_change = node.parent.parent
    # Check siblings of grandparent node and subtrees.
    elif node.parent.parent is not None and True in [len(get_text_nodes(s)) > 0 for s in
                                                     node.parent.parent.next_siblings]:
        for s in node.parent.parent.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break

    if node_to_change is not None:
        if isinstance(node_to_change, bs4.element.NavigableString):
            node_to_change.replace_with(f"{letter_img['alt']}{node_to_change.string.strip()}")
            logging.warning(f"Changed node text! New string: {letter_img['alt']}{node_to_change.string.strip()}")
        else:
            node_to_change.string = f"{letter_img['alt']}{node_to_change.string.strip()}"
            logging.warning(f"Changed node text! New string: {letter_img['alt']}{node_to_change.string.strip()}")
        return True

    return False


def strip_html(
        text, permitted_tags=tuple(PERMITTED_TAGS), tags_to_decompose=tuple(TAGS_TO_DECOMPOSE),
        strip_initial=False,
        prettify=True,
):
    """Decomposes all tags in tags_to_decompose and unwraps all other tags that
       are **not** in permitted_tags.
    """
    permitted_tags = set(permitted_tags)
    tags_to_decompose = set(tags_to_decompose)

    soup = BeautifulSoup(text, "html.parser")

    if strip_initial:
        # First look for initials/drop caps, extract the letter, and insert the letter into
        # the next node containing text.
        nodes = [soup]
        while len(nodes) > 0:
            curr = nodes.pop()
            if not isinstance(curr, bs4.element.Tag):
                continue
            if add_letter_from_initial_img(curr):
                logging.warning(f"Found initial.")
            nodes += list(curr.children)

    # Traverse the rest of the tree via BFS; remove or unwrap extraneous tags.
    nodes = [soup]
    # Use a while loop instead of a for loop because |nodes| is changing
    # as we traverse the tree.
    while len(nodes) > 0:
        curr = nodes.pop()
        i = 0
        while i < len(curr.contents):
            child = curr.contents[i]
            if child is None or child.name is None:
                i += 1
                continue
            if child.name in tags_to_decompose:
                child.decompose()
                i -= 1
            elif child.name not in permitted_tags:
                child.unwrap()
                i -= 1
            nodes.append(child)
            i += 1
    if prettify:
        return soup.prettify("utf-8")
    else:
        return soup
