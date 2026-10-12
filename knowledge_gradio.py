import gradio as gr
import uuid
import requests
from webui.Knowledge_UI import add, delete, query_list, update, upload_docs, delete_selected_file, get_kb_file_list, read_file_content, update_file_content
from webui.Chat_UI import chatai_to_backend
from webui.Summary_UI import summarize_conversation, export_note, AUTO_OPTION, NEW_OPTION
from webui.Quiz_UI import generate_quiz, save_quiz_notes
from configs.setting import BACKEND_URL

def on_app_load():
    # 生成一个唯一的ID
    return str(uuid.uuid4())


def _on_load():
    """页面加载钩子：为**每个浏览器会话**注入独立的 session_id，并刷新知识库下拉框。

    注意：不能用 `gr.State(value=on_app_load())`——那样 uuid 只在模块导入时求值一次，
    Gradio 会把同一个字符串拷贝给所有会话，导致不同用户共用 temp/data/<session_id>/ 目录。
    """
    return (on_app_load(),) + _refresh_kb_dropdowns()


def _refresh_kb_dropdowns():
    """统一刷新五个知识库下拉框：管理页 name_select / 文件管理页 kb_drop / 导出面板 kb_choice / 聊天页 kb_choice_chat / 出题面板 quiz_kb"""
    try:
        c = list(requests.get(f"{BACKEND_URL}/knowledge/queryall_knowledge", timeout=10).json() or [])
    except Exception:
        c = []
    first = c[0] if c else None
    return (
        gr.update(choices=c, value=first),
        gr.update(choices=c, value=first),
        gr.update(choices=c + [AUTO_OPTION, NEW_OPTION], value=AUTO_OPTION),
        gr.update(choices=[""] + c, value=""),
        gr.update(choices=c, value=None),
    )

custom_css = """
/* 强制知识库下拉菜单向下展开 */
#name_select_drop .choices {
    top: 100% !important;
    bottom: auto !important;
    max-height: 200px !important;   /* 限制最大高度，让内容滚动 */
    overflow-y: auto !important;    /* 出现滚动条，而不是改变方向 */
}
"""

with gr.Blocks(fill_width=True, fill_height=False) as demo:  # 将 fill_height 改为 False 更可靠
    # 不传初始值：真正的 session_id 由 _on_load() 在每次页面加载时注入（每个会话各不相同）
    session_state = gr.State()
    kbs = query_list()['choices']  # 知识库名称列表（用于各页知识库下拉）
    with gr.Tabs() as tabs:
        with gr.Tab("聊天机器人"):
            gr.Markdown("#" "聊天机器人")
            with gr.Row():
                with gr.Column(scale=1, variant="panel") as sidebar_left:
                    sys_prompt = gr.Textbox(label="系统提示词", value="You are a helpful assistant")
                    history_len = gr.Slider(minimum=1, maximum=10, value=1, label="保留历史消息的数量")
                    temperature = gr.Slider(minimum=0.01, maximum=2.0, value=0.5, step=0.01, label="temperature")
                    max_tokens = gr.Slider(minimum=64, maximum=512, value=512, step=8, label="max_tokens")
                    stream = gr.Checkbox(label="stream", value=True)
                    kb_choice_chat = gr.Dropdown(
                        label="指定知识库（留空自动按相关性匹配）",
                        choices=[""] + kbs,
                        value="",
                        interactive=True,
                    )
                    mode_radio = gr.Radio(["问问题", "出题目"], label="操作模式", value="问问题")
                    with gr.Column(visible=False) as quiz_panel:
                        quiz_kb = gr.Dropdown(
                            label="出题知识库",
                            choices=kbs,
                            value=None,
                            interactive=True,
                        )
                        quiz_num = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="题目数量")
                        quiz_type = gr.Dropdown(["混合", "选择题", "填空题", "判断题", "简答题"], value="混合", label="题型")
                        quiz_save_sel = gr.CheckboxGroup(
                            choices=["1", "2", "3", "4", "5"],
                            value=[],
                            label="选择要保存的题目（对完答案后勾选错题）",
                        )
                        quiz_btn = gr.Button("开始出题", variant="primary")
                        save_quiz_btn = gr.Button("保存选中题目为笔记", variant="secondary")
                        save_quiz_status = gr.Textbox(label="保存状态", lines=2, interactive=False)
                with gr.Column(scale=10) as main:
                    chatbot = gr.Chatbot(height=500)
                    chat_iface = gr.ChatInterface(
                        fn=chatai_to_backend,
                        multimodal=True,
                        chatbot=chatbot,
                        additional_inputs=[
                            sys_prompt,
                            history_len,
                            temperature,
                            max_tokens,
                            stream,
                            session_state,  # 将唯一ID传递给聊天接口
                            kb_choice_chat  # 指定知识库（留空自动匹配）
                        ]
                    )
                    with gr.Row():
                        chat_summarize_btn = gr.Button("总结知识点", variant="primary")
                    quiz_output = gr.Textbox(label="出题结果", lines=16, visible=False, interactive=False)

        with gr.Tab("知识库管理"):
            gr.Markdown("#" "知识库管理")
            with gr.Row():
                name = gr.Textbox(label="知识库名称", placeholder="请输入名字")
            with gr.Row():
                description = gr.Textbox(label="知识库描述", placeholder="请输入介绍")
            with gr.Row():
                message_box = gr.Markdown(visible=False)
            with gr.Row():
                create_button = gr.Button("创建知识库")
                delete_button = gr.Button("删除知识库")

            with gr.Row():
                name_select = gr.Dropdown(
                    label="选择知识库",
                    choices=kbs,
                )
            with gr.Row():
                file_uploaded = gr.Files(label='已选择文件（支持多选/分次累积，可点右侧 × 移除）', visible=False)
            with gr.Row():
                file_upload = gr.Files(label='选择文件')
            with gr.Row():
                chunk_size = gr.Number(label='知识库中单段文本最大长度', value=128, minimum=1, maximum= 800)
                chunk_overlap = gr.Number(label='知识库中相邻文本重合长度', value=20, minimum=0, maximum=400)
            with gr.Row():
                upload_button = gr.Button("上传文件并向量化")
            with gr.Row():
                upload_message_box = gr.Markdown(visible=False)




        with gr.Tab('知识库文件管理'):
            with gr.Row():
                kb_drop = gr.Dropdown(
                    label="选择知识库",
                    choices=kbs,
                    interactive=True
                )
            with gr.Row():
                file_drop = gr.Dropdown(
                    label="该知识库下文件列表",
                    choices=[],
                    interactive=True
                )
            with gr.Row():
                view_btn = gr.Button("查看文件内容", variant="secondary")
                save_file_btn = gr.Button("保存修改", variant="primary")
                del_btn = gr.Button("删除文件", variant="stop")
                refresh_btn = gr.Button("刷新文件列表", variant="secondary")
            file_content = gr.Textbox(label="文件内容（可编辑，修改后点击保存）", lines=18, interactive=True, visible=True)
            status_text = gr.Textbox(label="操作状态", interactive=False)

        with gr.Tab("对话问答整理"):
            gr.Markdown("### 对话问答笔记整理器")
            gr.Markdown("在「聊天机器人」页面点击「总结知识点」整理本次问答笔记；选择导出后，可手动选择知识库分类保存（默认自动匹配，匹配不到则自动新建）。")
            with gr.Row():
                export_radio = gr.Radio(["是", "否"], label="是否导出为文档？", value="否")
                clear_note_btn = gr.Button("清空笔记")
            note_output = gr.Textbox(label="对话问答笔记（可编辑，修改后点击导出）", lines=22, interactive=True)
            with gr.Column(visible=False) as export_panel:
                with gr.Row():
                    note_name = gr.Textbox(label="设置笔记名称", value="", placeholder="便于分类复习，建议填写；留空则自动命名。⚠️ 同名笔记将追加到原笔记并自动标注追加日期")
                with gr.Row():
                    kb_choice = gr.Dropdown(
                        label="保存到知识库",
                        choices=kbs + [AUTO_OPTION, NEW_OPTION],
                        value=AUTO_OPTION,
                        interactive=True,
                    )
                    new_kb_name = gr.Textbox(label="新建知识库名称", placeholder="选择「⚡新建知识库」时填写", visible=False)
                with gr.Row():
                    export_btn = gr.Button("导出并保存到知识库", variant="primary")
                export_status = gr.Markdown(value="", visible=True)

    demo.load(
        fn=_on_load,
        outputs=[session_state, name_select, kb_drop, kb_choice, kb_choice_chat, quiz_kb],
        queue=True
    )

    create_button.click(add, [name, description], [message_box, name, description]).then(fn=_refresh_kb_dropdowns, outputs=[name_select, kb_drop, kb_choice, kb_choice_chat, quiz_kb])
    delete_button.click(delete, [name_select], [message_box, name]).then(fn=_refresh_kb_dropdowns, outputs=[name_select, kb_drop, kb_choice, kb_choice_chat, quiz_kb])

    file_upload.upload(fn=update, inputs=[file_upload], outputs=[file_uploaded, file_upload])

    upload_button.click(upload_docs, [name_select, file_uploaded, chunk_size, chunk_overlap], [upload_message_box, file_uploaded]).then(
        fn=get_kb_file_list,
        inputs=[name_select],
        outputs=[file_drop]
    )

    kb_drop.change(
    fn=get_kb_file_list,
    inputs=[kb_drop],
    outputs=[file_drop]
    )

    refresh_btn.click(
        fn=get_kb_file_list,
        inputs=[kb_drop],
        outputs=[file_drop]
    )

    del_btn.click(
        fn=delete_selected_file,
        inputs=[kb_drop, file_drop],
        outputs=[status_text, file_drop]
    ).then(
        fn=get_kb_file_list,
        inputs=[kb_drop],
        outputs=[file_drop]
    )

    # 查看文件（笔记）内容，用于复习梳理
    view_btn.click(
        fn=read_file_content,
        inputs=[kb_drop, file_drop],
        outputs=[file_content]
    )

    # 修改文件（笔记）内容：写回文件并重建向量索引
    save_file_btn.click(
        fn=update_file_content,
        inputs=[kb_drop, file_drop, file_content],
        outputs=[status_text]
    )

    # ================= 对话问答整理 =================
    # 是否导出为文档：是 → 展开保存面板；否 → 收起
    export_radio.change(
        fn=lambda v: gr.update(visible=(v == "是")),
        inputs=[export_radio],
        outputs=[export_panel]
    )
    # 选择「⚡新建知识库」时显示新库名称输入框
    kb_choice.change(
        fn=lambda v: gr.update(visible=(v == NEW_OPTION)),
        inputs=[kb_choice],
        outputs=[new_kb_name]
    )
    # 导出并保存到知识库（可选择自动匹配 / 已有知识库 / 手动新建，便于分类归档复习）
    export_btn.click(
        fn=export_note,
        inputs=[note_output, note_name, kb_choice, new_kb_name],
        outputs=[export_status]
    ).then(
        # 导出后自动刷新五个知识库下拉框，让新建/更新后的知识库立即可见
        fn=_refresh_kb_dropdowns,
        outputs=[name_select, kb_drop, kb_choice, kb_choice_chat, quiz_kb]
    )
    # 清空笔记与面板
    clear_note_btn.click(
        fn=lambda: (gr.update(value=""), gr.update(value="否"), gr.update(visible=False),
                    gr.update(value=""), gr.update(value=AUTO_OPTION), gr.update(visible=False), gr.update(value="")),
        outputs=[note_output, export_radio, export_panel, note_name, kb_choice, new_kb_name, export_status]
    )

    # ================= 聊天机器人页「总结知识点」按钮 =================
    # 点击后执行整理流程 → 清空当前对话（显示 + 内部历史 State）→ 跳转到「对话问答整理」标签页（索引 3）
    chat_summarize_btn.click(
        fn=summarize_conversation,
        inputs=[chatbot],
        outputs=[note_output, export_panel, export_status]
    ).then(
        # 清空聊天界面显示
        fn=lambda: [],
        outputs=[chatbot]
    ).then(
        # 同步清空 ChatInterface 内部历史 State，确保下一次提问是全新对话
        fn=lambda x: (x, x),
        inputs=[chatbot],
        outputs=[chat_iface.chatbot_state, chat_iface.chatbot_value]
    ).then(
        fn=lambda: gr.Tabs(selected=3),
        outputs=[tabs]
    )

    # ================= 操作模式：问问题 / 出题目 =================
    # 切换「出题目」时显示出题面板与结果区；「问问题」时隐藏
    mode_radio.change(
        fn=lambda v: (
            gr.update(visible=(v == "出题目")),
            gr.update(visible=(v == "出题目")),
        ),
        inputs=[mode_radio],
        outputs=[quiz_panel, quiz_output]
    )
    # 开始出题：调后端 /knowledge/quiz 生成题目并展示（只出题，不保存）
    quiz_btn.click(
        fn=generate_quiz,
        inputs=[quiz_kb, quiz_num, quiz_type],
        outputs=[quiz_output]
    )
    # 对完答案后：保存勾选的题目为笔记（自动命名「错题」，同名自动追加）
    save_quiz_btn.click(
        fn=save_quiz_notes,
        inputs=[quiz_output, quiz_kb, quiz_save_sel],
        outputs=[save_quiz_status]
    )
    # 题目数量变化时联动「选择题目」多选项（默认不勾选，由用户对完答案后选）
    quiz_num.change(
        fn=lambda n: gr.update(
            choices=[str(i) for i in range(1, int(n or 1) + 1)],
            value=[],
        ),
        inputs=[quiz_num],
        outputs=[quiz_save_sel],
    )
if __name__ == "__main__":
    # server_name="0.0.0.0"：监听容器所有网卡，Docker 端口映射（7860->7860）才能从宿主机访问
    # 加 __main__ 保护：便于测试/其他入口 import 本模块而不真的启动服务
    demo.launch(css=custom_css, server_name="0.0.0.0")
