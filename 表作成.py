import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="LaTeX表作成ツール", layout="wide")

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def resize_dataframe(df, target_rows, target_cols):
    current_rows, current_cols = df.shape

    # 行の調整
    if target_rows < current_rows:
        df = df.iloc[:target_rows, :]
    elif target_rows > current_rows:
        rows_to_add = target_rows - current_rows
        new_rows = pd.DataFrame([[""] * current_cols] * rows_to_add, columns=df.columns)
        df = pd.concat([df, new_rows], ignore_index=True)

    # 列の調整
    current_rows, current_cols = df.shape 
    if target_cols < current_cols:
        df = df.iloc[:, :target_cols]
    elif target_cols > current_cols:
        for i in range(current_cols, target_cols):
            new_col = str(i) 
            while new_col in df.columns:
                new_col = new_col + "_"
            df[new_col] = ""
    
    # --- 列名・行名を 1始まり にリセット ---
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
        st.session_state.column_format_input = fmt + fmt[-1] * (len(new_df.columns) - len(fmt))
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
    """
    結合セルごとに異なる色を割り当てて見やすくする
    """
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # プレビュー用のカラーパレット (パステルカラーで文字は黒)
    colors = [
        'background-color: #85C1E9; color: black;',  # 水色
        'background-color: #82E0AA; color: black;',  # 緑
        'background-color: #F7DC6F; color: black;',  # 黄色
        'background-color: #F1948A; color: black;',  # 赤/ピンク
        'background-color: #BB8FCE; color: black;',  # 紫
        'background-color: #E59866; color: black;',  # オレンジ
    ]

    if "merge_list" in st.session_state:
        for idx, m in enumerate(st.session_state.merge_list):
            r, c, rs, cs = m["r"], m["c"], m["rs"], m["cs"]
            
            # 順番に色を回す
            current_style = colors[idx % len(colors)]
            
            if r < df.shape[0] and c < df.shape[1]:
                 style_df.iloc[r:r+rs, c:c+cs] = current_style
    return style_df

# ---------------------------------------------------------
# LaTeX 生成ロジック
# ---------------------------------------------------------

def generate_custom_latex(df, merges, caption, label, col_fmt, use_booktabs, center):
    rows, cols = df.shape
    skip_horizontal = np.zeros((rows, cols), dtype=bool)

    top = "\\toprule" if use_booktabs else "\\hline"
    mid = "\\midrule" if use_booktabs else "\\hline"
    bottom = "\\bottomrule" if use_booktabs else "\\hline"

    lines = []
    lines.append("\\begin{table}[htbp]")
    if center:
        lines.append("  \\centering")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")

    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(f"    {top}")

    for i in range(rows):
        row_cells = []
        for j in range(cols):
            if skip_horizontal[i, j]:
                continue

            # 結合開始地点か判定
            is_merge_start = False
            current_merge = None
            for m in merges:
                if m['r'] == i and m['c'] == j:
                    is_merge_start = True
                    current_merge = m
                    break
            
            val = df.iloc[i, j]
            text = str(val) if val is not None else ""

            if is_merge_start:
                rs, cs = current_merge['rs'], current_merge['cs']
                
                # 横方向のスキップ予約
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
                # 縦結合の下側に隠れているかチェック
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

def add_merge():
    # 1始まり(UI) -> 0始まり(内部)
    st.session_state.merge_list.append({
        "r": st.session_state.merge_r_input - 1,
        "c": st.session_state.merge_c_input - 1,
        "rs": st.session_state.merge_rs_input,
        "cs": st.session_state.merge_cs_input
    })
def remove_merge(i):
    st.session_state.merge_list.pop(i)


# =========================================================
# UI レイアウト
# =========================================================

st.title("LaTeX表作成ツール")

col_left, col_right = st.columns([1, 1.5], gap="large")

# --- 【左カラム】 設定エリア ---
with col_left:
    st.subheader("設定・構造", divider="gray")

    # 1. サイズ変更
    st.markdown("##### 1. サイズ変更")
    s1, s2 = st.columns(2)
    with s1:
        st.number_input("行数 (Rows)", min_value=1, key="rows_input", on_change=on_shape_change)
    with s2:
        st.number_input("列数 (Cols)", min_value=1, key="cols_input", on_change=on_shape_change)

    st.divider()

    # 2. 結合設定
    st.markdown("##### 2. セル結合 (Merge)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.number_input("開始行", 1, max(1, st.session_state.rows_input), 1, key="merge_r_input")
    with m2: st.number_input("開始列", 1, max(1, st.session_state.cols_input), 1, key="merge_c_input")
    with m3: st.number_input("高さ", 1, 20, 1, key="merge_rs_input")
    with m4: st.number_input("幅", 1, 20, 1, key="merge_cs_input")
    
    if st.button("結合を追加", use_container_width=True, on_click=add_merge):
        pass

    st.caption("▼ 結合プレビュー")
    st.dataframe(
        st.session_state.df.style.apply(lambda _: highlight_merges(st.session_state.df), axis=None),
        use_container_width=True,
        height=150
    )

    if st.session_state.merge_list:
        with st.expander("現在の結合リスト", expanded=False):
            for idx, m in enumerate(st.session_state.merge_list):
                c_txt, c_btn = st.columns([4, 1])
                c_txt.text(f"No.{idx+1}: 行{m['r']+1}, 列{m['c']+1} から {m['rs']}x{m['cs']}")
                c_btn.button("✕", key=f"del_{idx}", on_click=remove_merge, args=(idx,))

    st.divider()

    # 3. オプション
    st.markdown("##### 3. 出力オプション")
    use_booktabs = st.checkbox("Booktabs (きれいな罫線)", value=True)
    center_table = st.checkbox("Center (中央揃え)", value=True)
    o1, o2 = st.columns(2)
    caption = o1.text_input("Caption", "表のキャプション")
    label = o2.text_input("Label", "tab:mytable")

    if "column_format_input" not in st.session_state:
        st.session_state.column_format_input = "c" * len(st.session_state.df.columns)
    column_format = st.text_input("列フォーマット (例: ccc)", key="column_format_input")


# --- 【右カラム】 編集・出力エリア ---
with col_right:
    st.subheader("データ編集 & 出力", divider="blue")
    
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="fixed",
        use_container_width=True,
        key="main_editor",
        height=400,
        hide_index=False
    )

    st.write("") 
    if st.button("LaTeXコードを生成", type="primary", use_container_width=True):
        st.session_state.df = edited_df
        st.session_state.df.index = np.arange(1, len(st.session_state.df) + 1)

        try:
            latex_code = generate_custom_latex(
                st.session_state.df,
                st.session_state.merge_list,
                caption,
                label,
                column_format,
                use_booktabs,
                center_table
            )
            
            st.success("生成完了")
            st.code(latex_code, language="latex")
            
            pkgs = []
            if use_booktabs: pkgs.append(r"\usepackage{booktabs}")
            if st.session_state.merge_list: pkgs.append(r"\usepackage{multirow}")
            if pkgs:
                st.caption(f"必要なパッケージ: " + ", ".join(pkgs))

        except Exception as e:
            st.error(f"エラー: {e}")