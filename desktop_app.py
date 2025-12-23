import flet as ft
import asyncio
import os
import sys
import pyperclip
import time
import psutil
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional
from v9 import OptimizedRAGService, OptimizedConfig, Answer, Chunk

# Ensure Windows compatibility for async and subprocess
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class ResearchAssistantApp:
    def __init__(self):
        self.config = OptimizedConfig()
        self.service = OptimizedRAGService(self.config)
        self.page: Optional[ft.Page] = None
        
        # UI Components
        self.chat_history = ft.ListView(expand=True, spacing=15, padding=20, auto_scroll=True)
        self.query_input = ft.TextField(
            hint_text="Ask a research question...",
            expand=True,
            on_submit=self.handle_query,
            border=ft.InputBorder.NONE,
            bgcolor=ft.Colors.TRANSPARENT,
            color=ft.Colors.WHITE,
            cursor_color=ft.Colors.BLUE_ACCENT_400,
            text_size=16,
        )
        self.project_list = ft.ListView(expand=True, spacing=10)
        self.status_text = ft.Text("System Standby", size=12, color=ft.Colors.BLUE_GREY_400, italic=True)
        self.loading_ring = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False, color=ft.Colors.BLUE_ACCENT_400)
        
        # Performance Monitors
        self.cpu_usage = ft.Text("CPU Load: 0%", size=11, color=ft.Colors.GREY_400)
        self.ram_usage = ft.Text("Memory: 0%", size=11, color=ft.Colors.GREY_400)
        self.active_project_title = ft.Text("RESEARCH ASSISTANT", size=18, weight=ft.FontWeight.BOLD)
        
        # Doc List
        self.doc_list_view = ft.ListView(expand=True, spacing=10, padding=10)
        
        # State
        self.selected_project_name = None
        
        # Panels (Will be initialized in main)
        self.sidebar = None
        self.main_content_area = None
        self.right_panel = None
        self.chat_ui = None
        self.empty_ui = None

    async def main(self, page: ft.Page):
        self.page = page
        page.title = "RESEARCH ASSISTANT | Advanced Research Intelligence"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 0
        page.spacing = 0
        page.window.width = 1350
        page.window.height = 900
        page.bgcolor = "#0B0E14"  # Deep Nordic Navy

        page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE_ACCENT)

        # 1. SIDEBAR
        self.sidebar = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ft.Colors.BLUE_ACCENT_400, size=28),
                            ft.Text("RESEARCH", size=18, weight=ft.FontWeight.BOLD),
                        ]),
                        ft.Text("STUDIO", size=11, color=ft.Colors.BLUE_ACCENT_200, weight=ft.FontWeight.W_300),
                    ], spacing=2),
                    padding=ft.padding.only(bottom=30)
                ),
                ft.Text("KNOWLEDGE HUBS", size=10, color=ft.Colors.GREY_600, weight=ft.FontWeight.BOLD),
                ft.Container(content=self.project_list, expand=True, padding=ft.padding.symmetric(vertical=10)),
                ft.ElevatedButton(
                    "New Hub",
                    icon=ft.Icons.ADD_ROUNDED,
                    on_click=self.show_create_project_dialog,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_ACCENT_700, color=ft.Colors.WHITE, shape=ft.RoundedRectangleBorder(radius=8)),
                    width=220,
                ),
                ft.Divider(height=40, color=ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
                ft.Column([
                    ft.Row([ft.Icon(ft.Icons.SPEED_ROUNDED, size=14, color=ft.Colors.BLUE_ACCENT_700), self.cpu_usage]),
                    ft.Row([ft.Icon(ft.Icons.MEMORY_ROUNDED, size=14, color=ft.Colors.BLUE_ACCENT_700), self.ram_usage]),
                ], spacing=10),
                ft.Container(height=10),
                ft.Row([self.loading_ring, self.status_text], spacing=10),
            ]),
            width=280,
            padding=30,
            gradient=ft.LinearGradient(begin=ft.alignment.top_center, end=ft.alignment.bottom_center, colors=["#141A24", "#0D1117"]),
            border=ft.border.only(right=ft.border.BorderSide(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE))),
        )

        # 2. CHAT INTERFACE
        header = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("CENTRAL INTELLIGENCE", size=10, color=ft.Colors.BLUE_ACCENT_400, weight=ft.FontWeight.BOLD),
                    self.active_project_title,
                ], spacing=2),
                ft.IconButton(ft.Icons.DELETE_SWEEP_ROUNDED, tooltip="Clear Interface", on_click=self.clear_chat, icon_color=ft.Colors.GREY_600),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.symmetric(horizontal=30, vertical=20),
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLACK),
            border=ft.border.only(bottom=ft.border.BorderSide(1, ft.Colors.with_opacity(0.05, ft.Colors.WHITE))),
        )

        input_bar = ft.Container(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ft.Colors.BLUE_ACCENT_700),
                    self.query_input,
                    ft.IconButton(ft.Icons.COPY_ROUNDED, tooltip="Clipboard Mode (//clipboard)", on_click=lambda _: self._append_tag("//clipboard"), icon_color=ft.Colors.BLUE_ACCENT_400, icon_size=18),
                    ft.IconButton(ft.Icons.LANGUAGE_ROUNDED, tooltip="Web Search Mode (//web)", on_click=lambda _: self._append_tag("//web"), icon_color=ft.Colors.BLUE_ACCENT_400, icon_size=18),
                    ft.IconButton(ft.Icons.SEND_ROUNDED, icon_color=ft.Colors.BLACK, bgcolor=ft.Colors.BLUE_ACCENT_400, on_click=self.handle_query, icon_size=18),
                ], spacing=15),
                padding=ft.padding.symmetric(horizontal=20, vertical=8),
                bgcolor="#1C212B",
                border_radius=12,
                border=ft.border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
            ),
            padding=ft.padding.only(left=40, right=40, bottom=30, top=10),
        )

        self.chat_ui = ft.Column([
            header,
            ft.Container(content=ft.SelectionArea(content=self.chat_history), expand=True),
            input_bar
        ], expand=True, spacing=0, visible=True)

        # 3. EMPTY UI
        self.empty_ui = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.AUTO_AWESOME, size=100, color=ft.Colors.with_opacity(0.05, ft.Colors.CYAN)),
                ft.Text("SYSTEM OFFLINE", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Select a Knowledge Hub to activate the research core", size=14, color=ft.Colors.GREY_700),
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            expand=True,
            visible=True,
        )

        self.main_content_area = ft.Container(
            content=self.empty_ui,
            expand=True,
            bgcolor="#0F131A"
        )

        # 4. RIGHT PANEL
        self.right_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.FOLDER_SPECIAL_ROUNDED, size=16, color=ft.Colors.CYAN_400),
                    ft.Text("SOURCE MANIFEST", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500),
                ], spacing=10),
                ft.Container(content=self.doc_list_view, expand=True, margin=ft.margin.only(top=10, bottom=20)),
                ft.ElevatedButton(
                    "Upload PDFs",
                    icon=ft.Icons.UPLOAD_ROUNDED,
                    on_click=self.pick_files,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_ACCENT), color=ft.Colors.BLUE_ACCENT_200, shape=ft.RoundedRectangleBorder(radius=8)),
                    width=210,
                )
            ]),
            width=260,
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
            border=ft.border.only(left=ft.border.BorderSide(1, ft.Colors.with_opacity(0.05, ft.Colors.WHITE))),
            visible=False # Hidden until project load
        )

        # Assembly
        page.add(
            ft.Row([
                self.sidebar,
                self.main_content_area,
                self.right_panel
            ], expand=True, spacing=0)
        )

        self.refresh_projects()
        page.run_task(self.update_system_stats)
        page.update()

    async def update_system_stats(self):
        while True:
            try:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                self.cpu_usage.value = f"CPU: {cpu}%"
                self.ram_usage.value = f"MEM: {mem}%"
                if self.page:
                    self.page.update()
                await asyncio.sleep(4)
            except:
                break

    def refresh_projects(self):
        projects = self.service.list_projects()
        self.project_list.controls.clear()
        for name in projects:
            is_selected = name == self.selected_project_name
            def create_delete_handler(p_name):
                return lambda _: self.confirm_delete_project(p_name)

            self.project_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Row([
                            ft.Icon(ft.Icons.HUB_ROUNDED if is_selected else ft.Icons.HUB_OUTLINED, size=18, color=ft.Colors.BLUE_ACCENT_400 if is_selected else ft.Colors.GREY_600),
                            ft.Text(name, size=14, color=ft.Colors.WHITE if is_selected else ft.Colors.GREY_400, weight=ft.FontWeight.BOLD if is_selected else ft.FontWeight.NORMAL),
                        ], spacing=12, expand=True),
                        ft.IconButton(
                            ft.Icons.DELETE_OUTLINE_ROUNDED,
                            icon_size=16,
                            icon_color=ft.Colors.GREY_700,
                            tooltip="Delete Hub",
                            on_click=create_delete_handler(name)
                        )
                    ], spacing=12, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.padding.only(left=15, right=5, top=8, bottom=8),
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_ACCENT) if is_selected else ft.Colors.TRANSPARENT,
                    on_click=self.on_project_click,
                    data=name,
                )
            )
        if self.page:
            self.page.update()

    def confirm_delete_project(self, name):
        def do_delete(e):
            self.service.delete_project(name)
            if self.selected_project_name == name:
                self.selected_project_name = None
                self.main_content_area.content = self.empty_ui
                self.right_panel.visible = False
                self.status_text.value = "Hub Declassified"
            
            dialog.open = False
            self.refresh_projects()
            self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("DELETION PROTOCOL", color=ft.Colors.RED_400, size=16, weight=ft.FontWeight.BOLD),
            content=ft.Text(f"Are you sure you want to permanently declassify and delete Hub: {name.upper()}? This action is irreversible.", size=14),
            actions=[
                ft.TextButton("ABORT", on_click=lambda e: self.close_overlay(dialog)),
                ft.ElevatedButton("CONFIRM DELETE", on_click=do_delete, bgcolor=ft.Colors.RED_900, color=ft.Colors.WHITE)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    async def on_project_click(self, e):
        await self.select_project(e.control.data)

    async def select_project(self, name):
        self.selected_project_name = name
        self.status_text.value = "Initializing Engine..."
        self.loading_ring.visible = True
        self.refresh_projects()
        self.page.update()

        loop = asyncio.get_running_loop()
        try:
            # We run loading in executor to keep UI responsive
            success = await loop.run_in_executor(None, self.service.load_project, name)
        except Exception as ex:
            self.status_text.value = "Engine Error"
            self.loading_ring.visible = False
            self.page.update()
            self.show_error_dialog(f"Engine Protocol Failure: {str(ex)}")
            # Even if engine fails, show the UI so user can try again or see documents
            success = True 

        self.loading_ring.visible = False
        if success:
            self.status_text.value = "Neural Link Stable"
            self.active_project_title.value = f"Hub: {name.upper()}"
            self.main_content_area.content = self.chat_ui
            self.right_panel.visible = True
            self.refresh_doc_list()
        else:
            self.status_text.value = "Link Interrupted"
            
        self.page.update()

    def refresh_doc_list(self):
        self.doc_list_view.controls.clear()
        if self.service.current_project:
            for path in self.service.current_project.pdf_paths:
                base_name = os.path.basename(path)
                self.doc_list_view.controls.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.ARTICLE_ROUNDED, size=14, color=ft.Colors.CYAN_700),
                            ft.Text(base_name, size=11, color=ft.Colors.GREY_400, overflow=ft.TextOverflow.ELLIPSIS),
                        ], spacing=10),
                        padding=8,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
                        border_radius=6,
                    )
                )
        self.page.update()

    async def pick_files(self, _):
        file_picker = ft.FilePicker(on_result=self.on_file_result)
        self.page.overlay.append(file_picker)
        self.page.update()
        file_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["pdf"])

    async def on_file_result(self, e: ft.FilePickerResultEvent):
        if not e.files or not self.selected_project_name:
            return
        
        for file in e.files:
            self.status_text.value = f"Ingesting {file.name}..."
            self.loading_ring.visible = True
            self.page.update()
            
            loop = asyncio.get_running_loop()
            try:
                success = await loop.run_in_executor(None, self.service.add_pdf_to_project, file.path)
                if success:
                    self.status_text.value = "Core Knowledge Expanded"
                else:
                    self.status_text.value = "Ingestion Partial"
            except Exception as ex:
                self.show_error_dialog(f"Ingestion Error: {str(ex)}")
                self.status_text.value = "Ingestion Failed"
            
            self.loading_ring.visible = False
            self.refresh_doc_list()
            self.page.update()

    def _append_tag(self, tag):
        current = self.query_input.value or ""
        if tag not in current:
            self.query_input.value = f"{tag} {current}".strip()
            self.page.update()

    async def clear_chat(self, _):
        self.chat_history.controls.clear()
        self.page.update()

    async def handle_query(self, _):
        raw_query = self.query_input.value.strip()
        if not raw_query or not self.selected_project_name:
            return
        
        # Detect Modes
        mode = "project"
        ext_context = None
        display_query = raw_query
        
        if "//clipboard" in raw_query:
            mode = "clipboard"
            try:
                # Force a small wait to ensure OS has released clipboard if just copied
                await asyncio.sleep(0.1)
                ext_context = pyperclip.paste()
                if not ext_context or not ext_context.strip():
                    self.show_error_dialog("Clipboard appears empty. Please copy some text first.")
                    return
                # Sanitize and log
                ext_context = ext_context.strip()
                print(f"[DEBUG] Clipboard fetched: {ext_context[:50]}...")
            except Exception as e:
                self.show_error_dialog(f"Clipboard Access Error: {str(e)}")
                return
            display_query = raw_query.replace("//clipboard", "").strip()
        elif "//web" in raw_query:
            mode = "web"
            display_query = raw_query.replace("//web", "").strip()
        
        if not display_query:
            display_query = raw_query # Fallback if only tag was sent
            
        self.query_input.value = ""
        self.chat_history.controls.append(
            ft.Row([
                ft.Container(
                    content=ft.Text(raw_query, color=ft.Colors.WHITE, size=15),
                    padding=16, bgcolor=ft.Colors.CYAN_900, border_radius=ft.border_radius.only(top_left=15, top_right=15, bottom_left=15),
                    width=600, # Use fixed width as fallback for max_width
                )
            ], alignment=ft.MainAxisAlignment.END)
        )
        
        bot_msg_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.AUTO_AWESOME, size=18, color=ft.Colors.CYAN_400),
                    ft.Text("RESEARCH CORE", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_400),
                    ft.Text(f"({mode.upper()} MODE)", size=9, color=ft.Colors.CYAN_700, weight=ft.FontWeight.BOLD),
                    ft.ProgressRing(width=12, height=12, stroke_width=2, color=ft.Colors.CYAN_400),
                ], spacing=10),
                ft.Text(f"Analyzing {mode if mode != 'project' else 'Knowledge Hub'}...", italic=True, size=14, color=ft.Colors.GREY_500),
            ], spacing=10),
            padding=20, bgcolor="#1A1A1E", border_radius=ft.border_radius.only(top_left=15, top_right=15, bottom_right=15),
            width=700, # Use fixed width as fallback for max_width
        )
        self.chat_history.controls.append(ft.Row([bot_msg_container], alignment=ft.MainAxisAlignment.START))
        self.page.update()
        
        try:
            # Use updated service call with mode and external context
            answer = await self.service.ask_question(display_query, mode=mode, external_context=ext_context)
            bot_msg_container.content.controls.clear()
            
            async def copy_response(e):
                pyperclip.copy(answer.answer)
                self.status_text.value = "Response copied to clipboard"
                self.page.update()
                await asyncio.sleep(2)
                self.status_text.value = f"Confidence: {answer.confidence:.1%}"
                self.page.update()

            bot_msg_container.content.controls.extend([
                ft.Row([
                    ft.Row([
                        ft.Icon(ft.Icons.AUTO_AWESOME, size=18, color=ft.Colors.BLUE_ACCENT_400),
                        ft.Text("RESEARCH CORE", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_ACCENT_400),
                        ft.Text(f"{answer.processing_time:.2f}s", size=10, color=ft.Colors.GREY_600),
                    ], spacing=10, expand=True),
                    ft.IconButton(ft.Icons.COPY_ALL_ROUNDED, icon_size=15, icon_color=ft.Colors.GREY_600, tooltip="Copy to clipboard", on_click=copy_response),
                ]),
                ft.Markdown(answer.answer, selectable=True),
            ])
            if answer.sources:
                def create_source_handler(s_path):
                    return lambda _: (pyperclip.copy(s_path), setattr(self.status_text, "value", f"Path copied: {os.path.basename(s_path)}"), self.page.update())

                source_links = []
                for s in answer.sources[:3]:
                    source_name = s['source']
                    page_info = f" (p.{s['page']})" if s['page'] > 0 else ""
                    source_links.append(
                        ft.TextButton(
                            content=ft.Text(f"{source_name}{page_info}", size=10, color=ft.Colors.BLUE_ACCENT_200),
                            on_click=create_source_handler(source_name),
                            style=ft.ButtonStyle(
                                padding=ft.padding.symmetric(horizontal=8, vertical=4),
                                shape=ft.RoundedRectangleBorder(radius=6),
                                overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_ACCENT),
                            ),
                            tooltip="Click to copy source path"
                        )
                    )
                bot_msg_container.content.controls.append(ft.Row(source_links, spacing=0))
            self.status_text.value = f"Confidence: {answer.confidence:.1%}"
        except Exception as ex:
            bot_msg_container.content.controls.append(ft.Text(f"Synthesis Interrupted: {str(ex)}", color=ft.Colors.RED_400))
        
        self.page.update()

    def show_create_project_dialog(self, _):
        name_field = ft.TextField(label="Hub Name", border_color=ft.Colors.CYAN_900)
        domain_field = ft.Dropdown(label="Domain", options=[ft.dropdown.Option("General"), ft.dropdown.Option("Technical"), ft.dropdown.Option("Academic")], value="General")
        
        def close_dialog(e):
            dialog.open = False
            self.page.update()

        def create_and_close(e):
            if name_field.value:
                self.service.create_project(name_field.value, domain_field.value)
                self.refresh_projects()
                close_dialog(e)

        dialog = ft.AlertDialog(
            title=ft.Text("INITIALIZE HUB PROTOCOL", size=16, weight=ft.FontWeight.BOLD),
            content=ft.Column([ft.Text("Define parameters for the research environment.", size=13, color=ft.Colors.GREY_400), name_field, domain_field], tight=True, spacing=20),
            actions=[ft.TextButton("ABORT", on_click=close_dialog), ft.ElevatedButton("INITIALIZE", on_click=create_and_close, bgcolor=ft.Colors.CYAN_900, color=ft.Colors.WHITE)],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def show_error_dialog(self, message):
        dialog = ft.AlertDialog(
            title=ft.Text("SYSTEM ADVISORY", color=ft.Colors.RED_400, size=16, weight=ft.FontWeight.BOLD),
            content=ft.Text(message, size=14),
            actions=[ft.TextButton("ACKNOWLEDGE", on_click=lambda e: self.close_overlay(dialog))],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def close_overlay(self, overlay):
        overlay.open = False
        self.page.update()

if __name__ == "__main__":
    app = ResearchAssistantApp()
    ft.app(target=app.main)
