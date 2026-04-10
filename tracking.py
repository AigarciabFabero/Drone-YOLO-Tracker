import time
import cv2
import json
import base64
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from djitellopy import Tello


class YOLOTracker:
    """Gestor de tracking con YOLO"""
    
    def __init__(self, model, device: str = "cuda", conf_threshold: float = 0.5, 
                 classes_filter: Optional[List[int]] = None):
        """
        Inicializa el tracker
        
        Args:
            model: Modelo YOLO cargado
            device: Dispositivo ('cuda' o 'cpu')
            conf_threshold: Umbral de confianza mínimo
            classes_filter: Lista de IDs de clases a filtrar (None = todas)
        """
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.classes_filter = classes_filter
        self.last_inference_time = 0
        
    def track_image(self, image: np.ndarray) -> Tuple[List[Dict], float, np.ndarray]:
        """
        Ejecuta tracking en una imagen
        
        Args:
            image: Imagen numpy (BGR)
            
        Returns:
            Tupla (detecciones, tiempo_inferencia, resultados_yolo)
        """
        inicio_inferencia = time.time()
        
        # Ejecutar modelo con tracking
        resultados = self.model.track(
            image,
            device=self.device,
            half=True,
            conf=self.conf_threshold,
            classes=self.classes_filter,
            persist=True,
            verbose=False
        )
        
        fin_inferencia = time.time()
        self.last_inference_time = fin_inferencia - inicio_inferencia
        
        # Extraer detecciones como diccionarios
        detecciones = self._extract_detections(resultados)
        
        return detecciones, self.last_inference_time, resultados
    
    def _extract_detections(self, resultados) -> List[Dict]:
        """Extrae detecciones de los resultados de YOLO como diccionarios"""
        detecciones = []
        
        if resultados[0].boxes is not None:
            for box in resultados[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detecciones.append({
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "confidence": round(float(box.conf[0]), 2),
                    "class": int(box.cls[0]),
                    "id": int(box.id[0]) if box.id is not None else None
                })
        
        return detecciones
    
    def get_annotated_image(self, resultados) -> np.ndarray:
        """Retorna imagen anotada con bounding boxes"""
        return resultados[0].plot()


class TrackingJSON:
    """Gestor de serialización y guardado de tracking a JSON"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Inicializa el generador de JSON
        
        Args:
            output_dir: Directorio para guardar JSONs (por defecto directorio actual)
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def encode_image_base64(self, image: np.ndarray, quality: int = 80) -> str:
        """
        Codifica imagen a Base64 (JPG comprimido)
        
        Args:
            image: Imagen numpy (BGR)
            quality: Calidad JPEG (0-100)
            
        Returns:
            String Base64 de la imagen
        """
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg_as_txt = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_txt
    
    def create_payload(self, detecciones: List[Dict], image: np.ndarray, 
                      include_image: bool = True) -> Dict:
        """
        Crea el payload JSON con detecciones e imagen
        
        Args:
            detecciones: Lista de diccionarios de detecciones
            image: Imagen original (BGR)
            include_image: Si incluir imagen en Base64
            
        Returns:
            Diccionario con el payload
        """
        payload = {}
        
        if include_image:
            payload["frame"] = self.encode_image_base64(image)
        
        payload["detections"] = detecciones
        
        return payload
    
    def save_json(self, payload: Dict, filename: Optional[str] = None) -> str:
        """
        Guarda el payload como JSON
        
        Args:
            payload: Diccionario con los datos
            filename: Nombre del archivo (por defecto usa timestamp)
            
        Returns:
            Ruta del archivo guardado
        """
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"deteccion_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(payload, f)
        
        return str(filepath)
    
    def save_detections_only(self, detecciones: List[Dict], 
                            filename: Optional[str] = None) -> str:
        """
        Guarda solo las detecciones sin imagen (más ligero)
        
        Args:
            detecciones: Lista de diccionarios de detecciones
            filename: Nombre del archivo
            
        Returns:
            Ruta del archivo guardado
        """
        payload = {
            "detections": detecciones
        }
        
        return self.save_json(payload, filename)


class FlightController:
    """Controlador de dinámica de vuelo para seguir un target específico"""
    
    def __init__(self, tello: Tello, image_width: int = 960, image_height: int = 720, speed: int = 100):
        """
        Inicializa el controlador de vuelo
        
        Args:
            tello: Instancia del dron Tello
            image_width: Ancho de la imagen (píxeles)
            image_height: Alto de la imagen (píxeles)
            speed: Velocidad de movimiento (10-100)
        """
        self.tello = tello
        self.image_width = image_width
        self.image_height = image_height
        self.speed = max(20, min(speed, 100))
        self.k_rotate = 1.0 # [0.2 - 1.0] # ecomendado [0.7 - 1.0]
        self.k_up_down = 1.0 # [0.2 - 1.0] # Recomendado [0.7 - 1.0]
        self.k_forward_back = 0.8 # [0.2 - 1.0] Recomendado [0.5 - 0.8]
        self.tolerance__yy = 0.08 # Caja central
        self.tolerance__xx = 0.08 # Caja central
        self.tolerance_min_area = 0.15
        self.tolerance_max_area = 0.35 
        self.offset_cy = 0.35 # Cuanto más alto , más alejado de la cabeza. ¡Sistema de referencia esquina superior izquierda! 0.5 sería el centro centro.
        self.min_forward_speed = 20
        self.paused = False
        
        # Centro de la imagen
        self.center_x = image_width / 2
        self.center_y = image_height / 2
        
        # Zonas de tolerancia
        self.tolerance_x = image_width * self.tolerance__xx
        self.tolerance_y = image_height * self.tolerance__yy
        
        # Rango de distancia aceptable
        self.min_area = image_width * image_height * self.tolerance_min_area
        self.max_area = image_width * image_height * self.tolerance_max_area
        
    def get_target_detection(self, detecciones: List[Dict], target_id: int) -> Optional[Dict]:
        """Obtiene la detección del target por ID"""
        for det in detecciones:
            if det['id'] == target_id:
                return det
        return None
    
    def calculate_bbox_center(self, detection: Dict) -> Tuple[float, float]:
        """Calcula el centro del bounding box"""
        cx = (detection['x1'] + detection['x2']) * 0.5
        altura = (detection['y2'] - detection['y1'])
        cy = detection['y1'] + altura * self.offset_cy # ¡Sistema de referencia esquina superior izquierda!
        return cx, cy
    
    def calculate_bbox_area(self, detection: Dict) -> float:
        """Calcula el área del bounding box"""
        width = detection['x2'] - detection['x1']
        height = detection['y2'] - detection['y1']
        return width * height
    
    def calculate_commands(self, detection: Dict) -> Dict[str, int]:
        """
        Calcula los comandos de vuelo proporcionales para seguir el target.
        """
        commands = {
            'left_right': 0,        # Mantenemos en 0 para evitar que el dron vuele de lado
            'forward_backward': 0,  
            'up_down': 0,           
            'rotate': 0          
        }
        
        cx, cy = self.calculate_bbox_center(detection)
        area = self.calculate_bbox_area(detection)
        
        # 1. Rotación (YAW) - Gira sobre su propio eje
        dx = cx - self.center_x
        if abs(dx) > self.tolerance_x:
            # La velocidad de giro es proporcional a qué tan lejos está del centro
            rotate_speed = int(((dx / self.center_x) * self.speed) * self.k_rotate)
            # Clampeamos el valor para no exceder los límites de velocidad permitidos
            commands['rotate'] = max(-self.speed, min(self.speed, rotate_speed))
            
        # 2. Altitud (UP-DOWN) - Sube o baja proporcionalmente
        dy = cy - self.center_y
        if abs(dy) > self.tolerance_y:
            # Invertimos el signo: si dy es positivo (el target está abajo en la imagen), el dron debe bajar
            up_down_speed = int((-(dy / self.center_y) * self.speed) * self.k_up_down)
            commands['up_down'] = max(-self.speed, min(self.speed, up_down_speed))
            
        # 3. Profundidad Proporcional (FORWARD-BACKWARD) - Adiós al "latigazo"
        if area < self.min_area:
            # Está lejos. Avanzamos proporcionalmente.
            # Si el área se acerca a 0, la velocidad tiende a self.speed. Si se acerca a min_area, tiende a 0.
            fb_speed = int((((self.min_area - area) / self.min_area) * self.speed) * self.k_forward_back)
            # Aplicamos un mínimo (ej. 15) para que no se quede paralizado si la velocidad calculada es muy baja
            commands['forward_backward'] = max(self.min_forward_speed, min(self.speed, fb_speed))
            
        elif area > self.max_area:
            # Está muy cerca. Retrocedemos proporcionalmente.
            exceso_area = area - self.max_area
            fb_speed = int(-(exceso_area / self.max_area) * self.speed)
            # Aplicamos el mínimo negativo
            commands['forward_backward'] = max(-self.speed, min(-15, fb_speed))
            
        return commands
    
    def send_commands(self, commands: Dict[str, int]) -> None:
        """Envía los comandos al dron"""
        lr = commands['left_right']
        fb = commands['forward_backward']
        ud = commands['up_down']
        yaw = commands['rotate']
        
        if self.paused:
            print("self.tello.send_rc_control(0, 0, 0, 0)")
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        else:
            try:
                print(f"self.tello.send_rc_control({lr}, {fb}, {ud}, {yaw})")
                self.tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as e:
                print(f"[ERROR] {e}")

    def toggle_pause(self) -> None:
        """Pausa/Reaunuda el vuelo"""
        self.paused = not self.paused
        if self.paused:
            print("self.tello.send_rc_control(0, 0, 0, 0)")
            self.tello.send_rc_control(0, 0, 0, 0)
    
    def follow_target(self, detecciones: List[Dict], target_id: int) -> Optional[Dict]:
        """
        Hace que el dron siga un target específico
        
        Args:
            detecciones: Lista de detecciones YOLO
            target_id: ID del target a seguir
            
        Returns:
            Detección del target o None si no existe
        """
        target = self.get_target_detection(detecciones, target_id)
        
        if target is None:
            self.tello.send_rc_control(0, 0, 0, 0)
            return None
        
        commands = self.calculate_commands(target)
        self.send_commands(commands)
        
        return target
    
    def stop(self) -> None:
        """Detiene todos los movimientos"""
        self.tello.send_rc_control(0, 0, 0, 0)
    

    def draw_guide(self, img, target_detection):

        if target_detection is None:
            return img

        # 1. Parámetros de la imagen y centro (Referencia Fija)
        h, w, _ = img.shape
        cx_img, cy_img = w // 2, h // 2

        # --- DIBUJAR REFERENCIAS ESTÁTICAS ---

        # Cruz Azul (Centro de la imagen)
        color_blue = (255, 0, 0)
        cv2.line(img, (cx_img - 25, cy_img), (cx_img + 25, cy_img), color_blue, 2)
        cv2.line(img, (cx_img, cy_img - 25), (cx_img, cy_img + 25), color_blue, 2)

        # Zona de Tolerancia (Rectángulo Verde basado en tus variables)
        pt1 = (int(cx_img - self.tolerance_x), int(cy_img - self.tolerance_y))
        pt2 = (int(cx_img + self.tolerance_x), int(cy_img + self.tolerance_y))
        cv2.rectangle(img, 
                      (pt1), 
                      (pt2), 
                      (0, 255, 0), 2)

        # --- DIBUJAR DINÁMICA DEL TARGET ---

        if target_detection is not None:
            # Extraer posición del target
            tx, ty = self.calculate_bbox_center(target_detection)
            area = self.calculate_bbox_area(target_detection)

            # Punto Amarillo (Centro del Target)
            cv2.circle(img, (int(tx), int(ty)), 6, (0, 255, 255), -1)

            # Líneas Rojas de Error (Visualizan dx y dy de tu lógica)
            # dx: Error horizontal -> Activará 'rotate' (Yaw)
            cv2.line(img, (cx_img, int(ty)), (int(tx), int(ty)), (0, 0, 255), 2)
            # dy: Error vertical -> Activará 'up_down'
            cv2.line(img, (cx_img, cy_img), (cx_img, int(ty)), (0, 0, 255), 2)

            # --- INDICADOR DE PROFUNDIDAD (Forward/Backward) ---
            status_text = "DENTRO DE RANGO"
            text_color = (0, 255, 0)

            if area < self.min_area:
                status_text = f"LEJOS: AVANZANDO (+{int(((self.min_area-area)/self.min_area)*100)}%)"
                text_color = (0, 255, 255) 
            elif area > self.max_area:
                status_text = "CERCA: RETROCEDIENDO"
                text_color = (0, 0, 255) 

            cv2.putText(img, status_text, (20, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        return img