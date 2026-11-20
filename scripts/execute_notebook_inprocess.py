"""In-process notebook execution and output renderer.

Executes each code cell within an isolated execution context, captures stdout,
stderr, pandas HTML/plain displays, and base64 PNG images, populating standard
Jupyter notebook cell outputs (nbformat v4) without requiring ZMQ socket bindings.
"""

import os
import sys
import io
import base64
import contextlib
import traceback
import nbformat as nbf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def execute_notebook(
    input_path: str = "notebooks/fraud_triage.ipynb",
    output_path: str = "notebooks/fraud_triage.ipynb",
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Execution namespace
    exec_globals = {
        "__name__": "__main__",
        "__file__": os.path.abspath(input_path),
    }

    execution_count = 1

    print(f"Executing {len(nb.cells)} cells in {input_path}...")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        cell.outputs = []
        cell.execution_count = execution_count

        # Filter out ipython magic commands if any
        cleaned_lines = []
        for line in source.splitlines():
            if line.strip().startswith("%"):
                cleaned_lines.append(f"# {line}")
            else:
                cleaned_lines.append(line)
        exec_code = "\n".join(cleaned_lines)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Custom display hook to capture IPython displays
        captured_displays = []

        def custom_display(*args, **kwargs):
            for arg in args:
                if hasattr(arg, "filename") and os.path.exists(arg.filename):
                    # It's an IPython Image
                    with open(arg.filename, "rb") as img_f:
                        b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                    captured_displays.append(
                        nbf.v4.new_output(
                            output_type="display_data",
                            data={"image/png": b64_data, "text/plain": f"<IPython.core.display.Image object>"},
                            metadata={},
                        )
                    )
                elif isinstance(arg, pd.DataFrame):
                    captured_displays.append(
                        nbf.v4.new_output(
                            output_type="display_data",
                            data={
                                "text/html": arg.to_html(classes="table table-striped", index=False),
                                "text/plain": arg.to_string(index=False),
                            },
                            metadata={},
                        )
                    )
                else:
                    stdout_buf.write(str(arg) + "\n")

        exec_globals["display"] = custom_display

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                # Check if there are unclosed matplotlib plots
                plt.close("all")
                exec(exec_code, exec_globals)

                # If the code produced a matplotlib figure, capture it
                if plt.get_fignums():
                    fig = plt.gcf()
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
                    img_buf.seek(0)
                    b64_data = base64.b64encode(img_buf.read()).decode("utf-8")
                    captured_displays.append(
                        nbf.v4.new_output(
                            output_type="display_data",
                            data={"image/png": b64_data, "text/plain": "<Figure size ...>"},
                            metadata={},
                        )
                    )
                    plt.close("all")

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"Error in cell {i} (execution_count {execution_count}):\n{err_msg}")
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="error",
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=err_msg.splitlines(),
                )
            )
            break

        # Attach captured displays
        cell.outputs.extend(captured_displays)

        # Attach stdout
        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=stdout_text,
                )
            )

        # Attach stderr
        stderr_text = stderr_buf.getvalue()
        if stderr_text:
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=stderr_text,
                )
            )

        execution_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully executed and saved notebook to {output_path} with {execution_count - 1} executed cells.")


if __name__ == "__main__":
    execute_notebook()
