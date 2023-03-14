import pyutils.io as io
from bs4 import BeautifulSoup


def old_strip_html(text):
    soup = BeautifulSoup("".join(text))
    return " ".join(soup.get_text().strip().split())


def get_clean_text(str_obj):
    if str_obj is None:
        return ""
    return " ".join(str(str_obj).strip().split())


def format_nice_text(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    p_list = soup.findAll('p')
    if len(p_list) == 0:
        # Fall-back for if we have no <p> tags to work off
        return " ".join(soup.get_text().strip().split())
    else:
        text_list = []
        header = get_clean_text(p_list[0].prev_sibling)
        if header:
            text_list.append(header)
        for p_elem in p_list:
            clean_p_text = get_clean_text(p_elem.get_text())
            if clean_p_text:
                text_list.append(clean_p_text)
            clean_p_suffix = get_clean_text(p_elem.next_sibling)
            if clean_p_suffix:
                text_list.append(clean_p_suffix)
        return "\n\n".join(text_list)


def process_file(input_path, output_path, strip_html=False):
    data = io.read_jsonl(input_path)
    out = []
    for row in data:
        i = 1
        if strip_html:
            context = format_nice_text("\n\n".join(row["article"]))
        else:
            context = row["article"]
        while True:
            if f"question{i}" not in row:
                break
            out.append({
                "context": "".join(context),
                "query": " " + row[f"question{i}"].strip(),
                "option_0": " " + row[f"question{i}option1"].strip(),
                "option_1": " " + row[f"question{i}option2"].strip(),
                "option_2": " " + row[f"question{i}option3"].strip(),
                "option_3": " " + row[f"question{i}option4"].strip(),
                "label": row[f"question{i}_gold_label"] - 1,
            })
            i += 1
    io.write_jsonl(out, output_path)
