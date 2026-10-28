import flet as ft
import cv2
import threading
from Core.Logica import VideoProcessor, BiomechanicalAnalyzer
from Modelos.modelos_datos import PerfilAtletaModel

def build_app(page: ft.Page):
    page.title = "BiomecanicSport - Mallas Faciales y Cinemática Corporal"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 850
    page.window_height = 720

    processor = VideoProcessor()

    # Formulario del Atleta
    tf_nombre = ft.TextField(label="Nombre", value="Carlos", width=190)
    tf_apellido = ft.TextField(label="Apellido", value="Mendoza", width=190)
    tf_edad = ft.TextField(label="Edad", value="21", width=90)
    tf_fecha_nac = ft.TextField(label="Fecha Nacimiento", value="15/08/2003", width=160)
    tf_peso = ft.TextField(label="Peso (kg)", value="72.0", width=100)
    tf_altura = ft.TextField(label="Altura (cm)", value="175", width=100)
    
    dd_deporte = ft.Dropdown(
        label="Deporte",
        value="Fútbol",
        width=190,
        options=[
            ft.dropdown.Option("Fútbol"),
            ft.dropdown.Option("Baloncesto"),
            ft.dropdown.Option("Atletismo"),
            ft.dropdown.Option("Ciclismo"),
            ft.dropdown.Option("Natación")
        ]
    )
    tf_posicion = ft.TextField(label="Especialidad / Posición", value="Velocista", width=190)

    estado_texto = ft.Text("Estado: Listo para iniciar", color="amber")

    def ejecutar_analisis_completo(e):
        try:
            atleta = PerfilAtletaModel(
                nombre=tf_nombre.value,
                apellido=tf_apellido.value,
                edad=int(tf_edad.value),
                fecha_nacimiento=tf_fecha_nac.value,
                peso_kg=float(tf_peso.value),
                altura_cm=float(tf_altura.value),
                deporte=dd_deporte.value,
                posicion_rol=tf_posicion.value
            )
            ruta, _ = BiomechanicalAnalyzer.generar_diagnostico_y_guardar(atleta)
            estado_texto.value = f"¡Expediente creado con éxito en:\n{ruta}"
            estado_texto.color = "green"
            page.update()
        except Exception as ex:
            estado_texto.value = f"Error: {str(ex)}"
            estado_texto.color = "red"
            page.update()

    def ejecucion_camara():
        try:
            while True:
                frame, exito = processor.obtener_frame_procesado()
                if not exito:
                    break
                cv2.imshow("BiomecanicSport - Visor de Malla y Cinemática", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            processor.liberar_camara()
            estado_texto.value = "Cámara finalizada correctamente."
            estado_texto.color = "amber"
            page.update()

    def iniciar_visor_camara(e):
        if not processor.iniciar_camara():
            estado_texto.value = "Error: No se detectó la cámara o falta el archivo .task"
            estado_texto.color = "red"
            page.update()
            return

        estado_texto.value = "Cámara activa con esquemas de cabeza y cuello (Presiona 'Q' en la ventana para salir)."
        estado_texto.color = "green"
        page.update()

        hilo = threading.Thread(target=ejecucion_camara, daemon=True)
        hilo.start()

    # Controles de Configuración y Visión
    chk_espejo = ft.Checkbox(label="Efecto Espejo", value=True, on_change=lambda e: setattr(processor.config, 'espejo', e.control.value))
    chk_grises = ft.Checkbox(label="Escala de Grises", value=False, on_change=lambda e: setattr(processor.config, 'escala_grises', e.control.value))
    
    # Checkboxes para habilitar mallas, telemetría y puntos del cuello
    chk_malla = ft.Checkbox(label="Mostrar Malla Facial 3D", value=True, on_change=lambda e: setattr(processor, 'mostrar_malla', e.control.value))
    chk_telemetria = ft.Checkbox(label="Telemetría de Boca e Inclinación de Cabeza", value=True, on_change=lambda e: setattr(processor, 'mostrar_telemetria', e.control.value))
    chk_cuello = ft.Checkbox(label="Rastreo y Eje Cervical (Cuello)", value=True, on_change=lambda e: setattr(processor, 'mostrar_cuello', e.control.value))

    tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(
                text="1. Registro y Perfil",
                content=ft.Container(
                    padding=20,
                    content=ft.Column([
                        ft.Text("Perfil Antropométrico del Deportista", size=18, weight="bold"),
                        ft.Row([tf_nombre, tf_apellido, tf_edad, tf_fecha_nac]),
                        ft.Row([tf_peso, tf_altura, dd_deporte, tf_posicion]),
                        ft.Divider(),
                        ft.FilledButton("Generar Expediente Técnico", on_click=ejecutar_analisis_completo)
                    ])
                )
            ),
            ft.Tab(
                text="2. Visor RA y Cinemática",
                content=ft.Container(
                    padding=20,
                    content=ft.Column([
                        ft.Text("Opciones de Procesamiento Facial y Corporal", size=18, weight="bold"),
                        ft.Row([chk_espejo, chk_grises]),
                        ft.Divider(),
                        ft.Text("Esquemas de Movimiento:", size=14, weight="bold", color="blue_accent"),
                        ft.Column([chk_malla, chk_telemetria, chk_cuello]),
                        ft.Container(height=10),
                        ft.FilledButton("Lanzar Cámara con Rastreo", on_click=iniciar_visor_camara)
                    ])
                )
            )
        ],
        expand=True
    )

    page.add(tabs, ft.Divider(), estado_texto)