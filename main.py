from tracking import YOLOTracker, TrackingJSON, FlightController
from djitellopy import Tello
import cv2
from ultralytics import YOLO
import torch
import threading
import time
import datetime

def keepalive_worker(tello_obj: Tello, stop_event: threading.Event) -> None:
    """Usa el objeto tello directamente para evitar conflictos de socket"""
    print("[KeepAlive] Hilo de seguridad iniciado.")

    while not stop_event.is_set():
        try:
            # Si el dron está conectado, enviamos el comando a través de la librería
            if tello_obj.is_flying: 
                tello_obj.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"[KeepAlive] Error: {e}")
        
        for _ in range(100):
            if stop_event.is_set(): 
                break
            time.sleep(0.1)

#--------------------------------------------------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

drone_ip = "10.116.225.100"
stop_safety_threads = threading.Event()

model = YOLO('.\\models\\yolo26n.pt')  

# Conectar con el dron
tello = Tello(host=drone_ip)
tello.connect()
print("Dron conectado")

# Configurar video
tello.set_video_fps(Tello.FPS_30)
tello.set_video_resolution(Tello.RESOLUTION_720P)
tello.streamon()
frame_read = tello.get_frame_read()

# Inicializar tracker
tracker = YOLOTracker(
    model=model,
    device=device,
    conf_threshold=0.5,
    classes_filter=[0, 24, 63]
)

# Inicializar gestor de JSON
json_manager = TrackingJSON(output_dir="./output")

# Inicializar controlador de vuelo
flight_controller = FlightController(
    tello=tello,
    image_width=960,
    image_height=720,
    speed=100
)

safety_thread = threading.Thread(
    target=keepalive_worker,
    args=(tello, stop_safety_threads), 
    daemon=True
)
safety_thread.start()

target_id = None
show_guide = True

tello.takeoff()
cv2.namedWindow("drone", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("drone", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

while True:
    frame = frame_read.frame
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    battery = tello.get_battery()
    cv2.putText(img, f"Bateria: {battery}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if battery < 20:
        print(f"Batería baja: {battery}%")
        break

    detecciones, inference_time, resultados = tracker.track_image(img)
    
    annotated_image = tracker.get_annotated_image(resultados)
    
    info_text = "Detecciones:"
    for i, det in enumerate(detecciones):
        info_text += f" ID:{det['id']}"
    cv2.putText(annotated_image, info_text, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if target_id is not None:
        cv2.putText(annotated_image, f"[SIGUIENDO] Target ID: {target_id}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        target_detection = flight_controller.follow_target(detecciones, target_id)      
        if show_guide:
            annotated_image = flight_controller.draw_guide(annotated_image, target_detection)
        
        if target_detection is None:
            cv2.putText(annotated_image, f"Target ID {target_id} NO ENCONTRADO!", (20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_image, "Sin target seleccionado", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(annotated_image, f"Tiempo de inferencia: {inference_time:.4f}s", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    payload = json_manager.create_payload(detecciones, img, include_image=True)
    json_manager.save_json(payload, filename="deteccion_salida.json")
    
    cv2.imshow("drone", annotated_image)
    
    key = cv2.waitKey(1) & 0xff

    if key == 27:  # ESC
        print("Salida por ESC")
        break
    elif key == ord('p'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"foto_limpia_{timestamp}.jpg"
        cv2.imwrite(filename, img) 
        print(f"[FOTO] Captura limpia guardada como: {filename}")
    elif key == ord(' '):
        flight_controller.toggle_pause()
        estado = "pausado" if flight_controller.paused else "reanudado"
        print(f"[CONTROL] Vuelo {estado}")
    elif key == ord('b'):
        show_guide = not show_guide
        estado_guia = "Activada" if show_guide else "Desactivada"
        print(f"[UI] Guía visual {estado_guia}")
    elif ord('0') <= key <= ord('9'):  
        num = int(chr(key))
        if num == 0:
            target_id = None
            flight_controller.stop()
            print("[CONTROL] Seguimiento detenido")
        else:
            target_id = num
            print(f"[CONTROL] Target seleccionado: {target_id}")

# Liberar recursos
print("Cerrando conexión...")
stop_safety_threads.set()
flight_controller.stop()
tello.streamoff()
tello.end()
tello.land()
cv2.destroyAllWindows()
print("Finalizado")