import os
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Modelos.modelos_datos import CameraConfigModel, PerfilAtletaModel

class VideoProcessor:
    """Procesador avanzado de video para rastreo facial, cuello y cinemática corporal."""
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.config = CameraConfigModel()
        
        # Interruptores de visualización para la interfaz
        self.mostrar_malla = True
        self.mostrar_telemetria = True
        self.mostrar_cuello = True
        
        self.ruta_modelo = 'face_landmarker.task'
        self.detector = None
        self._inicializar_detector()

    def _inicializar_detector(self):
        if os.path.exists(self.ruta_modelo):
            base_options = python.BaseOptions(model_asset_path=self.ruta_modelo)
            opciones = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.FaceLandmarker.create_from_options(opciones)
        else:
            print(f"[ADVERTENCIA]: No se encontró '{self.ruta_modelo}'.")

    def iniciar_camara(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def obtener_frame_procesado(self):
        if not self.cap or not self.cap.isOpened():
            return None, False

        ret, frame = self.cap.read()
        if not ret:
            return None, False

        if self.config.espejo:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        if self.detector and self.mostrar_malla:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
            
            mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = self.detector.detect(mp_imagen)

            if resultado.face_landmarks:
                puntos = resultado.face_landmarks[0]
                
                # 1. Renderizar puntos del rostro
                for lm in puntos:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                if self.mostrar_telemetria:
                    # Telemetría de Apertura Bucal (IDs 13 y 14)
                    x_sup, y_sup = int(puntos[13].x * w), int(puntos[13].y * h)
                    x_inf, y_inf = int(puntos[14].x * w), int(puntos[14].y * h)
                    cv2.line(frame, (x_sup, y_sup), (x_inf, y_inf), (255, 0, 255), 2)
                    dist_boca = math.hypot(x_inf - x_sup, y_inf - y_sup)

                    # Telemetría de Inclinación de la Cabeza (Ojo Izq: 33, Ojo Der: 263)
                    x_izq, y_izq = int(puntos[33].x * w), int(puntos[33].y * h)
                    x_der, y_der = int(puntos[263].x * w), int(puntos[263].y * h)
                    cv2.line(frame, (x_izq, y_izq), (x_der, y_der), (255, 255, 0), 2)
                    
                    delta_x = x_der - x_izq
                    delta_y = y_der - y_izq
                    angulo_cabeza = math.degrees(math.atan2(delta_y, delta_x))

                    # HUD de Textos en Pantalla
                    cv2.putText(frame, f"Boca: {int(dist_boca)}px", (25, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if dist_boca < 25 else (0, 0, 255), 2)
                    cv2.putText(frame, f"Inclinacion Cabeza: {angulo_cabeza:.1f} deg", (25, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if self.mostrar_cuello:
                    # Estimación de la zona cervical / base del cuello a partir del mentón (ID 152) y barbilla inferior
                    # Esto permite empezar a trazar puntos de movimiento hacia el cuello en la malla facial
                    x_menton, y_menton = int(puntos[152].x * w), int(puntos[152].y * h)
                    # Proyectamos un punto estimado de la base del cuello hacia abajo verticalmente
                    x_cuello, y_cuello = x_menton, y_menton + int(h * 0.12)
                    
                    cv2.circle(frame, (x_menton, y_menton), 5, (0, 165, 255), -1)
                    cv2.circle(frame, (x_cuello, y_cuello), 7, (0, 165, 255), -1)
                    cv2.line(frame, (x_menton, y_menton), (x_cuello, y_cuello), (0, 165, 255), 3)
                    cv2.putText(frame, "Eje Cervical (Cuello)", (x_cuello + 10, y_cuello), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if self.config.escala_grises:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, True

    def liberar_camara(self):
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()


class BiomechanicalAnalyzer:
    @staticmethod
    def generar_diagnostico_y_guardar(atleta: PerfilAtletaModel):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nombre_carpeta = f"{atleta.nombre}_{atleta.apellido}".replace(" ", "_").lower()
        ruta_carpeta = os.path.join("Data", "Deportistas", nombre_carpeta)
        os.makedirs(ruta_carpeta, exist_ok=True)

        tiempo = np.linspace(0, 10, 50)
        angulo = 120 + 20 * np.sin(tiempo)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(tiempo, angulo, color='tab:blue', lw=2)
        ax.set_title(f"Rendimiento Articular - {atleta.nombre} {atleta.apellido}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Ángulo (°)")
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        ruta_grafica = os.path.join(ruta_carpeta, "grafica_rendimiento.png")
        plt.savefig(ruta_grafica, dpi=150)
        plt.close(fig)

        ruta_txt = os.path.join(ruta_carpeta, "reporte.txt")
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"=== REPORTE BIOMECÁNICO ===\n")
            f.write(f"Atleta: {atleta.nombre} {atleta.apellido}\n")
            f.write(f"Deporte: {atleta.deporte} ({atleta.posicion_rol})\n")

        return ruta_carpeta, ruta_grafica