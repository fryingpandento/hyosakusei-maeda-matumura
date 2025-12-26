import streamlit as st
import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------
# 1. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="LaTeX表作成ツール", layout="wide")

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def escape_latex(text):
    """LaTeXの特殊文字をエスケープする"""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\\': r'\textbackslash{}', 
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

import unicodedata

def range_to_coords(range_str):
    """ 'A1:B2' 形式の文字列を (r, c, rs, cs) の辞書に変換 """
    try:
        # 全角→半角変換 (NFKC正規化)
        range_str = unicodedata.normalize('NFKC', range_str)
        range_str = range_str.upper().strip()
        if ":" in range_str:
            start, end = range_str.split(':')
        else:
            start = end = range_str 
        
        def cell_to_idx(cell):
            match = re.match(r"([A-Z]+)([0-9]+)", cell)
            if not match: return None
            col_str, row_str = match.groups()
            row = int(row_str) - 1
            col = 0
            for char in col_str:
                col = col * 26 + (ord(char) - ord('A') + 1)
            col -= 1
            return row, col

        pos1 = cell_to_idx(start)
        pos2 = cell_to_idx(end)
        if pos1 is None or pos2 is None: return None

        r1, c1 = pos1
        r2, c2 = pos2
        return {
            "r": min(r1, r2),
            "c": min(c1, c2),
            "rs": abs(r1 - r2) + 1,
            "cs": abs(c1 - c2) + 1
        }
    except Exception:
        return None

def resize_dataframe(df, target_rows, target_cols):
    current_rows, current_cols = df.shape
    if target_rows < current_rows:
        df = df.iloc[:target_rows, :]
    elif target_rows > current_rows:
        rows_to_add = target_rows - current_rows
        new_rows = pd.DataFrame([[""] * current_cols] * rows_to_add, columns=df.columns)
        df = pd.concat([df, new_rows], ignore_index=True)

    current_rows, current_cols = df.shape 
    if target_cols < current_cols:
        df = df.iloc[:, :target_cols]
    elif target_cols > current_cols:
        for i in range(current_cols, target_cols):
            new_col = str(i) 
            while new_col in df.columns:
                new_col = new_col + "_"
            df[new_col] = ""
    
    df.columns = [str(i + 1) for i in range(df.shape[1])]
    df.index = np.arange(1, len(df) + 1)
    return df

def clean_merges(merges, rows, cols):
    valid = []
    for m in merges:
        if m["r"] + m["rs"] <= rows and m["c"] + m["cs"] <= cols:
            valid.append(m)
    return valid

def on_shape_change():
    if "main_editor" in st.session_state and isinstance(st.session_state["main_editor"], pd.DataFrame):
        base_df = st.session_state["main_editor"]
    else:
        base_df = st.session_state.df

    new_df = resize_dataframe(base_df, st.session_state.rows_input, st.session_state.cols_input)

    fmt = st.session_state.get("column_format_input", "c" * len(new_df.columns))
    if len(fmt) < len(new_df.columns):
        fill_char = fmt[-1] if fmt else "c"
        st.session_state.column_format_input = fmt + fill_char * (len(new_df.columns) - len(fmt))
    else:
        st.session_state.column_format_input = fmt[:len(new_df.columns)]

    st.session_state.df = new_df
    if "merge_list" in st.session_state:
        st.session_state.merge_list = clean_merges(
            st.session_state.merge_list,
            st.session_state.rows_input,
            st.session_state.cols_input
        )
    if "main_editor" in st.session_state:
        del st.session_state["main_editor"]

def highlight_merges(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    colors = [
        'background-color: #85C1E9; color: black;', 
        'background-color: #82E0AA; color: black;', 
        'background-color: #F7DC6F; color: black;', 
        'background-color: #F1948A; color: black;', 
        'background-color: #BB8FCE; color: black;', 
        'background-color: #E59866; color: black;', 
    ]
    if "merge_list" in st.session_state:
        for idx, m in enumerate(st.session_state.merge_list):
            r, c, rs, cs = m["r"], m["c"], m["rs"], m["cs"]
            current_style = colors[idx % len(colors)]
            if r < df.shape[0] and c < df.shape[1]:
                 style_df.iloc[r:r+rs, c:c+cs] = current_style
    return style_df

def generate_custom_latex(df, merges, caption, label, col_fmt, use_booktabs, center):
    rows, cols = df.shape
    skip_horizontal = np.zeros((rows, cols), dtype=bool)
    top = "\\toprule" if use_booktabs else "\\hline"
    mid = "\\midrule" if use_booktabs else "\\hline"
    bottom = "\\bottomrule" if use_booktabs else "\\hline"

    lines = []
    lines.append("\\begin{table}[htbp]")
    if center: lines.append("  \\centering")
    if caption: lines.append(f"  \\caption{{{caption}}}")
    if label: lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(f"    {top}")

    for i in range(rows):
        row_cells = []
        for j in range(cols):
            if skip_horizontal[i, j]: continue

            is_merge_start = False
            current_merge = None
            for m in merges:
                if m['r'] == i and m['c'] == j:
                    is_merge_start = True
                    current_merge = m
                    break
            
            val = df.iloc[i, j]
            text = escape_latex(str(val)) if val is not None and str(val) != "nan" else ""

            if is_merge_start:
                rs, cs = current_merge['rs'], current_merge['cs']
                for c_off in range(1, cs):
                    skip_horizontal[i, j + c_off] = True
                
                if rs > 1 and cs > 1:
                    cell = "\\multicolumn{" + str(cs) + "}{c}{\\multirow{" + str(rs) + "}{*}{" + text + "}}"
                elif rs > 1:
                    cell = "\\multirow{" + str(rs) + "}{*}{" + text + "}"
                elif cs > 1:
                    cell = "\\multicolumn{" + str(cs) + "}{c}{" + text + "}"
                else:
                    cell = text
                row_cells.append(cell)
            else:
                is_vertically_covered = False
                for m in merges:
                    if m['r'] < i < m['r'] + m['rs'] and m['c'] <= j < m['c'] + m['cs']:
                        is_vertically_covered = True
                        break
                if is_vertically_covered:
                    row_cells.append("") 
                else:
                    row_cells.append(text)
        
        lines.append("    " + " & ".join(row_cells) + " \\\\")
        if i == 0 and rows > 1:
            lines.append(f"    {mid}")

    lines.append(f"    {bottom}")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ---------------------------------------------------------
# State初期化
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.full((3, 3), ""))
    st.session_state.df.columns = ["1", "2", "3"]
    st.session_state.df.index = [1, 2, 3]

if "merge_list" not in st.session_state:
    st.session_state.merge_list = []
if "rows_input" not in st.session_state:
    st.session_state.rows_input = len(st.session_state.df)
if "cols_input" not in st.session_state:
    st.session_state.cols_input = len(st.session_state.df.columns)

def remove_merge(i):
    st.session_state.merge_list.pop(i)
def add_merge_by_range():
    coords = range_to_coords(st.session_state.range_input_val)
    if coords:
        st.session_state.merge_list.append(coords)
        st.session_state.range_input_val = "" 
    else:
        st.error("形式が無効です。例: A1:B2")

# =========================================================
# UI レイアウト
# =========================================================
st.title("LaTeX表作成ツール")

col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.subheader("設定・構造", divider="gray")
    st.markdown("##### 1. サイズ変更")
    s1, s2 = st.columns(2)
    with s1:
        st.number_input("行数 (Rows)", min_value=1, key="rows_input", on_change=on_shape_change)
    with s2:
        st.number_input("列数 (Cols)", min_value=1, key="cols_input", on_change=on_shape_change)

    st.divider()

    st.markdown("##### 2. セル結合 (Merge)")
    r1, r2 = st.columns([3, 1])
    with r1:
        st.text_input("範囲入力 (例: A1:B2)", key="range_input_val", placeholder="A1:B2")
    with r2:
        st.write("") 
        st.write("")
        st.button("追加", on_click=add_merge_by_range, use_container_width=True)

    st.caption("▼ 結合プレビュー")
    st.dataframe(
        st.session_state.df.style.apply(lambda _: highlight_merges(st.session_state.df), axis=None),
        use_container_width=True,
        height=150
    )
    if st.session_state.merge_list:
        with st.expander("結合リストの管理", expanded=False):
            for idx, m in enumerate(st.session_state.merge_list):
                c_txt, c_btn = st.columns([4, 1])
                c_txt.text(f"No.{idx+1}: ({m['r']+1}, {m['c']+1}) size {m['rs']}x{m['cs']}")
                c_btn.button("✕", key=f"del_{idx}", on_click=remove_merge, args=(idx,))

    st.divider()

    st.markdown("##### 3. 出力オプション")
    use_booktabs = st.checkbox("Booktabs (きれいな罫線)", value=True)
    center_table = st.checkbox("Center (中央揃え)", value=True)
    o1, o2 = st.columns(2)
    caption = o1.text_input("Caption", "表のキャプション")
    label = o2.text_input("Label", "tab:mytable")
    column_format = st.text_input("列フォーマット (例: ccc)", key="column_format_input")

with col_right:
    st.subheader("データ編集 & 出力", divider="blue")
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode

    # AgGridの設定
    gb = GridOptionsBuilder.from_dataframe(st.session_state.df)
    gb.configure_default_column(editable=True, groupable=True)
    
    # IME問題対策: シングルクリックで編集開始するように設定
    gb.configure_grid_options(singleClickEdit=True, stopEditingWhenCellsLoseFocus=True)
    
    gridOptions = gb.build()

    grid_response = AgGrid(
        st.session_state.df,
        gridOptions=gridOptions,
        height=400,
        width='100%',
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False, # 列幅の自動調整（好みで変更可）
        key='aggrid_table'
    )

    # データ更新の反映
    updated_df = grid_response['data']
    
    # 行・列数が変わった場合（AgGrid上では行追加は難しいので、リセット検知等）
    # シンプルにデータ変更があればsession_stateを更新
    if not updated_df.equals(st.session_state.df):
        st.session_state.df = updated_df
        # 行数・列数inputとの同期は、このツールでは「input -> shape change」の流れが主なので、
        # ここではdfの中身だけ更新しておく


    st.write("") 
    if st.button("LaTeXコードを生成", type="primary", use_container_width=True):
        st.session_state.df = edited_df
        st.session_state.df.index = np.arange(1, len(st.session_state.df) + 1)
        try:
            final_col_fmt = column_format if column_format else "c" * len(st.session_state.df.columns)
            latex_code = generate_custom_latex(
                st.session_state.df,
                st.session_state.merge_list,
                caption,
                label,
                final_col_fmt,
                use_booktabs,
                center_table
            )
            st.success("生成完了！")
            st.code(latex_code, language="latex")
            pkgs = []
            if use_booktabs: pkgs.append(r"\usepackage{booktabs}")
            if st.session_state.merge_list: pkgs.append(r"\usepackage{multirow}")
            if pkgs:
                st.info(f"必要なパッケージ: " + ", ".join(pkgs))
        except Exception as e:
            st.error(f"エラー: {e}")