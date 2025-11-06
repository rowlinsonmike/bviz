import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from diagrams import Diagram, Edge
from diagrams.generic.blank import Blank
from diagrams.aws.ml import Bedrock
from diagrams.onprem.client import User
from diagrams.onprem.client import Client
from diagrams.custom import Custom
from reportlab.lib.enums import TA_LEFT
import traceback
import os
import typer
from rich import print
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4  # or LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from PIL import Image as PILImage

def generate_pdf(
    name: str,
    image_path: str,
    text: str,
    output_path: str = "output.pdf",
    page_size=A4,
    full_bleed: bool = False  # set True for edge-to-edge image (no left/right margins)
):
    """
    Create a PDF with:
      - a PNG scaled to the full content width at the top (preserving aspect ratio),
      - the provided text below (auto-wrapped and paginated).

    Args:
        image_path: Path to the PNG image.
        text: Text content as a single string (newlines will be respected).
        output_path: Destination PDF path.
        page_size: ReportLab page size (e.g., A4, LETTER).
        full_bleed: If True, left/right margins are 0 to make the image span the entire page width.
    """
    # Margins
    if full_bleed:
        left_margin = right_margin = 0
        top_margin = bottom_margin = 15 * mm  # small top/bottom padding to avoid printer clipping
    else:
        # Nice, readable margins
        left_margin = right_margin = 20 * mm
        top_margin = bottom_margin = 20 * mm

    # Setup the document (Platypus)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=page_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    # Available width/height inside the frame
    frame_width = doc.width
    frame_height = doc.height

    # Load image to compute scaled dimensions preserving aspect ratio
    with PILImage.open(image_path) as img:
        img_w_px, img_h_px = img.size

    # Scale to fit within the frame (full width, capped by frame height)
    # Note: We use the pixel aspect ratio; ReportLab uses points for drawing, but ratio is unitless.
    width_target = frame_width
    height_target_from_width = width_target * (img_h_px / img_w_px)

    # Cap by frame height if needed so the image fits on the first page
    if height_target_from_width > frame_height:
        # Fit to height instead
        height_target = frame_height
        width_target = height_target * (img_w_px / img_h_px)
    else:
        height_target = height_target_from_width

    # Build the story
    styles = getSampleStyleSheet()
    # Optional: tweak paragraph formatting
    body_style = styles["Normal"]
    # Slightly increase leading for readability
    body_style = ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
    )
    title_style = ParagraphStyle(
        name='Title',
        fontSize=16,
        leading=22,
        alignment=TA_LEFT,
        textColor=(0.2, 0.2, 0.2),
        backgroundColor=(0.9, 0.9, 0.9),
        fontName='Helvetica-Bold'
    )

    story = []
    story.append(Paragraph(name, title_style))
    story.append(Spacer(1, 12))
    # Add the image (scaled)
    story.append(Image(image_path, width=width_target, height=height_target))

    # Small space between image and text
    story.append(Spacer(1, 8))

    # Render your text. Convert newlines to <br/> so Paragraph preserves line breaks.
    text_html = text.replace("\n", "<br/>")
    story.append(Paragraph(text_html, body_style))

    # Build the PDF
    doc.build(story)

### diagram
def wrap_label(s: str, width: int = 60) -> str:
    """
    Wrap text to a fixed width using '\\n' for Graphviz labels.
    Preserves paragraph breaks by splitting on double newlines.
    """
    if not s:
        return ""
    paragraphs = s.split("\n\n")
    wrapped_paras = []
    for para in paragraphs:
        lines = para.splitlines() if "\n" in para else [para]
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    replace_whitespace=False,
                )
                or [""]
            )
        wrapped_paras.append("\n".join(wrapped_lines))
    return "\n\n".join(wrapped_paras)

def wrap_legend_text(s: str, width: int = 80) -> str:
    """
    Wrap legend text to a wider width so it is more compact but still readable.
    """
    return wrap_label(s, width=width)

def content_to_text(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract text payload from a content item.
    Content items usually look like:
      - {"text": "some text"}
      - {"toolUse": {...}}
      - {"toolResult": {...}}
    """
    if "text" in item and isinstance(item["text"], str):
        return item["text"]
    return None

def extract_tool_use(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the inner toolUse dict if present, supporting a couple of key variants.
    """
    if "toolUse" in item and isinstance(item["toolUse"], dict):
        return item["toolUse"]
    if "tool_use" in item and isinstance(item["tool_use"], dict):
        return item["tool_use"]
    return None

def extract_tool_result(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the inner toolResult dict if present.
    Expected shape:
      {
        "toolUseId": "id",
        "content": [
          {"text": "..."},
          ...
        ]
      }
    """
    if "toolResult" in item and isinstance(item["toolResult"], dict):
        return item["toolResult"]
    if "tool_result" in item and isinstance(item["tool_result"], dict):
        return item["tool_result"]
    return None

def summarize_message_text(content_list: List[Dict[str, Any]]) -> str:
    """
    Combine all 'text' content items into one string for the message node/legend.
    """
    texts = []
    for item in content_list:
        t = content_to_text(item)
        if t:
            texts.append(t)
    return "\n".join(texts)

def format_tool_use_label_short(tool_use: Dict[str, Any], tu_id: str) -> str:
    """
    Short label for a tool-use node in the main graph.
    """
    tool_name = tool_use.get("toolName") or tool_use.get("name") or "unknown-tool"
    return f"<strong>{tu_id}</strong> TOOL CALL\n{tool_name}"

def format_tool_use_legend(tool_use: Dict[str, Any], tu_id: str) -> str:
    """
    Legend entry for a tool-use node, includes pretty input JSON and the raw toolUseId if present.
    """
    raw_id = tool_use.get("toolUseId") or tool_use.get("id") or "unknown-id"
    tool_name = tool_use.get("toolName") or tool_use.get("name") or "unknown-tool"
    input_payload = tool_use.get("input")
    try:
        pretty_input = wrap_legend_text(json.dumps(input_payload),width=80)
    except Exception:
        import traceback
        traceback.print_exc()
        pretty_input = str(input_payload)
    return f"<h1><strong>{tu_id} - Tool Call</strong></h1>\n<strong>- name:</strong> {tool_name}\n<strong>- toolUseId:</strong> {raw_id}\n<strong>- input:</strong>\n{pretty_input}"

def format_tool_result_label_short(tr_id: str) -> str:
    """
    Short label for a tool-result node in the main graph.
    """
    return f"{tr_id} TOOL RESULT"

def format_tool_result_legend(tool_result: Dict[str, Any], tr_id: str, tu_id: Optional[str]) -> str:
    """
    Legend entry for a tool-result node with captured text output.
    """
    raw_id = tool_result.get("toolUseId") or tool_result.get("tool_use_id") or "unknown-id"
    texts = []
    for c in tool_result.get("content", []):
        texts.append("<pre><code>{}</code></pre>".format(json.dumps(c,indent=4)[0:500]))
        # if "text" in c and isinstance(c["text"], str):
        #     texts.append(c["text"])
    joined = "\n".join(texts) if texts else ""
    tu_part = f" (for {tu_id})" if tu_id else ""
    return f"<strong>{tr_id} - Tool Result {tu_part}</strong>\n<strong>- toolUseId:</strong> {raw_id}\n<strong>- output:</strong>\n{joined}"

def visualize_bedrock_conversation(
    messages: List[Dict[str, Any]],
    out_path: str = "bedrock_conversation",
    out_format: str = "png",
    title: str = "Bedrock Conversation",
    compact_nodes: bool = True,
) -> str:
    """
    Visualize a Bedrock Converse API conversation using the diagrams library.

    - If compact_nodes=True (default), the main graph nodes show only IDs and short headers.
    - Renders toolUse nodes and toolResult nodes with numeric IDs (TU#, TR#).
    - Connects sequential messages.
    - Connects assistant -> toolUse (calls), toolUse -> toolResult (returns),
      and toolResult -> next assistant message (feeds).

    Args:
        messages: List of Bedrock Converse-style messages, e.g.:
            {
              "role": "user" | "assistant",
              "content": [
                {"text": "..."},
                {"toolUse": {"toolUseId": "id1", "toolName": "search", "input": {...}}},
                {"toolResult": {"toolUseId": "id1", "content": [{"text": "result"}]}}
              ]
            }
        out_path: Base filename (no extension) for the generated diagram.
        out_format: Diagram output format ("png", "svg", etc.).
        title: Diagram title.
        compact_nodes: If True, main graph nodes are short;

    Returns:
        The full path (including extension) to the generated diagram file.
    """
    graph_attr = {
        "splines": "spline",
        "rankdir": "LR",
        "fontsize": "12",
        "fontname": "Arial",
        "labelloc": "c",
        # Add margin (in inches) around the entire graph so PNG doesn’t clip
        "margin": "0.5",
        # Increase DPI for PNG rasterization so edges aren’t clipped by rounding
        "dpi": "200",
        # Optional: add more separation between nodes (horizontal/vertical)
        # These help when labels are long and the layout is tight
        "nodesep": "0.4",
        "ranksep": "0.6",
    }
    # Note: diagrams' Blank nodes render with an icon plus label; we keep consistent style
    node_attr = {
        "fontsize": "10",
        # Inner padding around labels inside nodes (horizontal,vertical in inches)
        "margin": "2,2",
        "fontname": "Arial",
        "shape": "box",
        "style": "rounded,filled",
        "color": "#555555",
        "fillcolor": "#F7F7F7",
    }

    # Storage for nodes and mapping
    message_nodes: List[Blank] = []
    tool_use_nodes_by_raw_id: Dict[str, Blank] = {}
    tool_use_short_id_by_raw_id: Dict[str, str] = {}
    tool_result_nodes_with_index: List[Tuple[int, Blank]] = []

    # Counters for numbering
    tu_counter = 0
    tr_counter = 0

    # Legend entries to the right
    legend_lines: List[str] = []

    with Diagram(
        title,
        filename=out_path,
        outformat=out_format,
        show=False,
        graph_attr=graph_attr,
        node_attr=node_attr,
    ) as _:
        # First pass: create nodes and collect legend info
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_list = msg.get("content", [])

            # Message numbering
            message_id = f"M{idx}"

            # Message text for legend (combined text blocks)
            message_text_full = summarize_message_text(content_list)

            # Build labels
            if compact_nodes:
                msg_label = f"{message_id} {role.upper()}"
            else:
                msg_text_wrapped = wrap_label(message_text_full, width=60)
                header = f"{message_id} {role.upper()}"
                msg_label = header if not msg_text_wrapped else f"{header}\n{msg_text_wrapped}"
            msg_node = Bedrock(msg_label) if "ASSISTANT" in msg_label else Custom(msg_label,str(Path(Path(__file__).parent / "icons/user.png")))
            message_nodes.append(msg_node)

            # Legend entry for message (always include full text)
            if message_text_full:
                legend_lines.append(f"<strong>{message_id} - {role.upper()} Message </strong>")
                legend_lines.append(f"{wrap_legend_text(message_text_full)}")
                legend_lines.append("")  # blank line separator

            # Create nodes for toolUse/toolResult items inside this message
            for item in content_list:
                tool_use = extract_tool_use(item)
                if tool_use:
                    tu_counter += 1
                    tu_id = f"TU{tu_counter}"
                    # Short label for the main node
                    tlabel = format_tool_use_label_short(tool_use, tu_id)
                    tlabel_wrapped = wrap_label(tlabel, width=60)
                    tnode = Custom(tlabel_wrapped,str(Path(Path(__file__).parent / "icons/tool.png")))

                    # Edge: message -> toolUse (call)
                    msg_node >> Edge(color="#8888FF", label="calls") >> tnode

                    # Remember mapping by toolUseId for linking to result
                    raw_tid = tool_use.get("toolUseId") or tool_use.get("id")
                    if isinstance(raw_tid, str):
                        tool_use_nodes_by_raw_id[raw_tid] = tnode
                        tool_use_short_id_by_raw_id[raw_tid] = tu_id

                    # Legend entry for tool call
                    legend_lines.append(format_tool_use_legend(tool_use, tu_id))
                    legend_lines.append("")
                    continue

                tool_result = extract_tool_result(item)
                if tool_result:
                    tr_counter += 1
                    tr_id = f"TR{tr_counter}"

                    # Short label for the main node
                    rlabel = format_tool_result_label_short(tr_id)
                    rlabel_wrapped = wrap_label(rlabel, width=60)
                    rnode = Custom(rlabel_wrapped, str(Path(Path(__file__).parent / "icons/response.png")))

                    # Link to toolUse if we can
                    raw_tid = tool_result.get("toolUseId") or tool_result.get("tool_use_id")
                    if isinstance(raw_tid, str) and raw_tid in tool_use_nodes_by_raw_id:
                        tool_use_nodes_by_raw_id[raw_tid] >> Edge(color="#55AA55", label="returns") >> rnode

                    # Also connect the current message to its contained tool result for clarity
                    msg_node >> Edge(color="#55AA55", style="dashed") >> rnode
                    tool_result_nodes_with_index.append((idx, rnode))

                    # Legend entry for tool result
                    tu_short = tool_use_short_id_by_raw_id.get(raw_tid)
                    legend_lines.append(format_tool_result_legend(tool_result, tr_id, tu_short))
                    legend_lines.append("")
                    continue

        # Second pass: sequential edges between message nodes
        for i in range(1, len(message_nodes)):
            prev_node = message_nodes[i - 1]
            curr_node = message_nodes[i]
            prev_role = (messages[i - 1].get("role") or "").lower()
            curr_role = (messages[i].get("role") or "").lower()
            color = "#999999"
            if prev_role == "user" and curr_role == "assistant":
                color = "#1F77B4"
            elif prev_role == "assistant" and curr_role == "user":
                color = "#FF7F0E"
            prev_node >> Edge(color=color) >> curr_node

        # Optional third pass: link toolResult -> next assistant message
        for result_idx, rnode in tool_result_nodes_with_index:
            # Find next assistant message after result_idx
            next_assistant_idx = None
            for j in range(result_idx + 1, len(messages)):
                if (messages[j].get("role") or "").lower() == "assistant":
                    next_assistant_idx = j
                    break
            if next_assistant_idx is not None:
                rnode >> Edge(color="#55AA55", style="dotted", label="feeds") >> message_nodes[next_assistant_idx]

    generate_pdf(title,f"{out_path}.png","\n".join(legend_lines),output_path=f"{title}.pdf",page_size=A4, full_bleed=False)
    return f"{out_path}.{out_format}"


app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context,path: str = typer.Option('', "--path", "-p", help="Path to Bedrock conversation in JSON file"),name: str = typer.Option('', "--name", "-n", help="Optional name for the diagram")):
    if not path and not name:
        return typer.echo(ctx.get_help())
    if not path.strip():
        print("[bold red]Must specify path to Bedrock conversation json file (-p)[/bold red]")
        return
    messages = json.loads(Path(Path(os.getcwd()) / path.strip()).read_text())
    name = name if name else "Bedrock Vision Diagram"
    output = name.lower().replace(' ','-')
    try:
        out = visualize_bedrock_conversation(
            messages,
            out_path=output,
            out_format="png",
            title=name,
            compact_nodes=True,
        )
        os.unlink("{}.png".format(output))
    except:
        print("[bold red]Error: [/bold red] {}").format(traceback.format_exc())
    print("[green]Success! [/green] Exported to {}.pdf".format(output))
