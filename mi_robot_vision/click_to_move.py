#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBVS Multi-color Teleoperado — xArm Lite 6 | Eye-in-Hand | ROS 2 Humble

Cambios en esta versión:
  · Opción A: offset fijo cámara→gripper (OFFSET_CAMARA_X/Y mm) aplicado
    justo antes de iniciar el descenso.
  · Opción C parcial: descenso en DOS FASES.
      1) Baja 50 % de la distancia total.
      2) Re-centra visualmente (SERVOING de nuevo, más preciso porque
         está más cerca del objeto).
      3) Aplica offset cámara→gripper otra vez.
      4) Baja el 50 % restante y agarra.
  · PID, Kalman, cola FIFO, gripper por topic, etc. se mantienen.
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
from xarm_msgs.srv import MoveCartesian, MoveJoint, SetInt16


# ══════════════════════════════════════════════════════════════════════════════
#  FILTRO KALMAN 1D
# ══════════════════════════════════════════════════════════════════════════════

class Kalman1D:
    def __init__(self, q_pos=2.0, q_vel=1.0, r=25.0):
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 500.0
        self.Q = np.array([[q_pos, 0.0], [0.0, q_vel]], dtype=np.float64)
        self.R = np.array([[r]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.initialized = False

    def reset(self):
        self.x[:] = 0.0
        self.P = np.eye(2, dtype=np.float64) * 500.0
        self.initialized = False

    def update(self, z: float, dt: float) -> float:
        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0
            self.initialized = True
            return float(self.x[0, 0])
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        z_vec = np.array([[z]], dtype=np.float64)
        y = z_vec - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROLADOR PID 1D
# ══════════════════════════════════════════════════════════════════════════════

class PID1D:
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 integral_max: float = 20.0,
                 d_filter_alpha: float = 0.3):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_max = integral_max
        self.alpha = d_filter_alpha
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self.initialized = False

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        if not self.initialized:
            self._prev_error = error
            self.initialized = True
            return self.Kp * error
        p_term = self.Kp * error
        self._integral += error * dt
        self._integral = float(np.clip(self._integral,
                                       -self.integral_max, self.integral_max))
        i_term = self.Ki * self._integral
        d_raw = (error - self._prev_error) / max(dt, 1e-3)
        self._d_filtered = (self.alpha * d_raw
                            + (1.0 - self.alpha) * self._d_filtered)
        d_term = self.Kd * self._d_filtered
        self._prev_error = error
        return p_term + i_term + d_term


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE COLOR
# ══════════════════════════════════════════════════════════════════════════════

def muestrear_hsv_parche(frame_bgr, x, y, tam=5):
    h, w = frame_bgr.shape[:2]
    r = tam // 2
    x0 = max(0, x - r); x1 = min(w, x + r + 1)
    y0 = max(0, y - r); y1 = min(h, y + r + 1)
    parche = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(parche, cv2.COLOR_BGR2HSV)
    return (int(np.median(hsv[:,:,0])),
            int(np.median(hsv[:,:,1])),
            int(np.median(hsv[:,:,2])))


def clasificar_color(h, s, v):
    if s < 50 or v < 50:  return "desconocido"
    if h < 10 or h > 170: return "rojo"
    if 10 <= h < 25:      return "naranja"
    if 25 <= h < 35:      return "amarillo"
    if 35 <= h < 85:      return "verde"
    if 85 <= h < 130:     return "azul"
    if 130 <= h < 160:    return "morado"
    return "desconocido"


# ══════════════════════════════════════════════════════════════════════════════
#  OBJETIVO COLOR
# ══════════════════════════════════════════════════════════════════════════════

class ObjetivoColor:
    def __init__(self, h, s, v, dh=10, ds=60, dv=60):
        self.h_clic, self.s_clic, self.v_clic = h, s, v
        self.etiqueta = clasificar_color(h, s, v)
        s_lo = max(40,  s - ds); s_hi = min(255, s + ds)
        v_lo = max(40,  v - dv); v_hi = min(255, v + dv)
        h_lo = h - dh; h_hi = h + dh
        self.rangos = []
        if h_lo < 0:
            self.rangos.append((np.array([0, s_lo, v_lo], np.uint8),
                                np.array([h_hi, s_hi, v_hi], np.uint8)))
            self.rangos.append((np.array([180+h_lo, s_lo, v_lo], np.uint8),
                                np.array([180, s_hi, v_hi], np.uint8)))
        elif h_hi > 180:
            self.rangos.append((np.array([h_lo, s_lo, v_lo], np.uint8),
                                np.array([180, s_hi, v_hi], np.uint8)))
            self.rangos.append((np.array([0, s_lo, v_lo], np.uint8),
                                np.array([h_hi-180, s_hi, v_hi], np.uint8)))
        else:
            self.rangos.append((np.array([h_lo, s_lo, v_lo], np.uint8),
                                np.array([h_hi, s_hi, v_hi], np.uint8)))
        hsv_pix = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0,0]
        self.bgr_chip = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def construir_mascara(self, hsv_img):
        m = None
        for lo, hi in self.rangos:
            mi = cv2.inRange(hsv_img, lo, hi)
            m = mi if m is None else (m | mi)
        return cv2.dilate(cv2.erode(m, None, iterations=2), None, iterations=2)


# ══════════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class PBVSMulticolorTeleop(Node):
    def __init__(self):
        super().__init__('pbvs_multicolor_teleop')

        # ── Suscripciones ──────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.subscription_esp = self.create_subscription(
            Float32, '/esp32/distancia_descenso', self.esp32_callback, 10)
        self.bridge = CvBridge()

        # ── Publicadores ───────────────────────────────────────────────────
        self.pub_descenso = self.create_publisher(
            Float32, '/esp32/distancia_descenso', 10)
        self.pub_gripper = self.create_publisher(
            String, '/lite6_gripper/command', 10)

        # ── Clientes de servicio ───────────────────────────────────────────
        self.cli_set_mode  = self.create_client(SetInt16,      '/ufactory/set_mode')
        self.cli_set_state = self.create_client(SetInt16,      '/ufactory/set_state')
        self.cli_joint     = self.create_client(MoveJoint,     '/ufactory/set_servo_angle')
        self.cli_cartesian = self.create_client(MoveCartesian, '/ufactory/set_position')

        # ── Configuración del robot ────────────────────────────────────────
        self.HOME_JOINTS_DEG = [0.0, -10.0, 35.0, 0.0, 55.0, 0.0]
        self.AREA_MIN_VALIDA = 400
        self.CENTRO_CAMARA   = (320, 380)
        self.SIGN_X    = -1.0
        self.SIGN_Y    = -1.0
        self.SWAP_AXES = False

        # ══════════════════════════════════════════════════════════════════
        # OFFSET CÁMARA → GRIPPER  (Opción A)
        # ══════════════════════════════════════════════════════════════════
        # Diferencia física en mm entre el eje óptico de la cámara y la
        # punta del gripper. Se aplica como movimiento relativo XY ANTES
        # de cada descenso (tanto fase 1 como fase 2).
        #
        # Convención (base del robot):
        #   +X = adelante,  +Y = izquierda
        #
        # Si la cámara está 2 cm adelante (+X) y 2 cm a la izquierda (+Y)
        # respecto al gripper, el gripper necesita moverse +20 en X y +20
        # en Y para quedar donde la cámara ve el centro. Ajusta signos
        # según tu montaje real.
        self.OFFSET_CAMARA_X =  11.0   # mm — positivo = cámara más adelante que gripper
        self.OFFSET_CAMARA_Y =  -12.0   # mm — positivo = cámara más a la izquierda que gripper

        # ══════════════════════════════════════════════════════════════════
        # DESCENSO EN DOS FASES  (Opción C parcial)
        # ══════════════════════════════════════════════════════════════════
        # La distancia total de descenso (d) se divide en dos mitades:
        #   Fase 1: baja  d * DESCENSO_FRACCION_1  (50%)
        #           → re-centra visualmente (SERVOING desde más cerca)
        #           → aplica offset cámara→gripper
        #   Fase 2: baja  d * (1 - DESCENSO_FRACCION_1)  (50% restante)
        #           → cierra gripper
        self.DESCENSO_FRACCION_1 = 0.5   # 50% en la primera bajada

        # ── PID ────────────────────────────────────────────────────────────
        self.Kp = 0.0017
        self.Ki = 0.001
        self.Kd = 0.003
        self.INTEGRAL_MAX   = 20.0
        self.D_FILTER_ALPHA = 0.3
        self.pid_x = PID1D(self.Kp, self.Ki, self.Kd,
                           self.INTEGRAL_MAX, self.D_FILTER_ALPHA)
        self.pid_y = PID1D(self.Kp, self.Ki, self.Kd,
                           self.INTEGRAL_MAX, self.D_FILTER_ALPHA)

        # ── Movimiento ─────────────────────────────────────────────────────
        self.STEP_MAX_XY = 8.0
        self.VEL_XY      = 30.0
        self.VEL_Z       = 25.0
        self.DEADZONE_PX         = 12.0
        self.TOLERANCIA_XY       = 25.0
        self.FRAMES_CENTRADO_MIN = 4
        self.GRIPPER_WAIT_SEC = 2.0
        self.GRIPPER_REPEAT   = 3
        self.ALTURA_LEVANTE          = 80.0
        self.FRAMES_PERDIDOS_MAX     = 8
        self.AREA_AGARRAR            = 15000
        self.FALLBACK_DESCENSO       = 200.0
        self.PUB_DESCENSO_HZ         = 5.0
        self.ESPERA_ANTES_CERRAR_SEC = 3.0

        # ── Zona de depósito ───────────────────────────────────────────────
        self.ZONA_X0    = 280.0
        self.ZONA_Y0    = 150.0
        self.ZONA_DX    =  70.0
        self.ZONA_DY    = -100.0
        self.ZONA_Z     = 200.0
        self.ZONA_RPY   = [math.pi, 0.0, 0.0]
        self.VEL_TRAVEL = 150.0
        ASIGNACION = {
            "rojo": (1,0), "naranja": (1,1), "amarillo": (1,2),
            "verde": (1,3), "azul": (0,0), "morado": (0,1),
            "desconocido": (0,2),
        }
        self.DROP_POSES = {
            c: [self.ZONA_X0 + f*self.ZONA_DX,
                self.ZONA_Y0 + co*self.ZONA_DY,
                self.ZONA_Z, *self.ZONA_RPY]
            for c, (f, co) in ASIGNACION.items()
        }

        # ── Estado interno ─────────────────────────────────────────────────
        self.cola_objetivos: list[ObjetivoColor] = []
        self.objetivo_actual: ObjetivoColor | None = None
        self.idx_actual        = 0
        self.permitir_arranque = False
        self.frames_perdidos   = 0
        self.frames_centrado   = 0
        self.kf_x = Kalman1D(); self.kf_y = Kalman1D()
        self._last_t = None
        self.distancia_esp32 = None
        self.ultimo_descenso_pick_mm = None
        self.movimiento_en_progreso = False
        self.gripper_en_progreso    = False
        self.gripper_timer_handle   = None
        self.fase_pick  = None
        self.estado     = "INICIALIZANDO"
        self.init_en_progreso = False
        self.target_pixel = None
        self.target_raw   = None
        self.area_objeto  = 0.0
        self.mask_global  = np.zeros((480, 640), dtype=np.uint8)
        self._pid_p_disp = 0.0
        self._pid_i_disp = 0.0
        self._pid_d_disp = 0.0

        # Control de la doble fase de descenso
        self._descenso_fase = 0          # 0=no iniciado, 1=bajó mitad, 2=bajó todo
        self._descenso_restante_mm = 0.0 # mm que faltan para la fase 2

        # ── OpenCV ─────────────────────────────────────────────────────────
        self.current_frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        self.last_raw_frame = None
        cv2.namedWindow("Control PBVS Multicolor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Control PBVS Multicolor", 900, 700)
        cv2.setMouseCallback("Control PBVS Multicolor", self._on_mouse)
        self._splash()
        cv2.imshow("Control PBVS Multicolor", self.current_frame)
        cv2.waitKey(1)

        # ── Timers ─────────────────────────────────────────────────────────
        self.create_timer(0.033, self.display_loop)
        self.create_timer(0.10,  self.control_loop)
        self.create_timer(1.0 / self.PUB_DESCENSO_HZ, self._publicar_descenso)
        self.timer_inicio = self.create_timer(1.0, self.iniciar_configuracion)

        self.get_logger().info(
            f"Nodo iniciado. PID Kp={self.Kp} Ki={self.Ki} Kd={self.Kd}  "
            f"Offset cam→grip X={self.OFFSET_CAMARA_X} Y={self.OFFSET_CAMARA_Y} mm  "
            f"Descenso 2 fases: {self.DESCENSO_FRACCION_1*100:.0f}%/{(1-self.DESCENSO_FRACCION_1)*100:.0f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLICADOR DESCENSO / GRIPPER
    # ══════════════════════════════════════════════════════════════════════

    def _publicar_descenso(self):
        msg = Float32(); msg.data = float(self.FALLBACK_DESCENSO)
        self.pub_descenso.publish(msg)

    def _gripper_send_raw(self, cmd):
        msg = String(); msg.data = cmd
        for _ in range(self.GRIPPER_REPEAT):
            self.pub_gripper.publish(msg)

    def _gripper_accion(self, cmd, on_done):
        if self.gripper_en_progreso:
            return
        if self.gripper_timer_handle is not None:
            try: self.gripper_timer_handle.cancel()
            except: pass
            self.gripper_timer_handle = None
        self.gripper_en_progreso = True
        self.get_logger().info(f"[GRIPPER] → {cmd!r}  (espera {self.GRIPPER_WAIT_SEC} s)")
        self._gripper_send_raw(cmd)
        _fired = [False]
        def _fin():
            if _fired[0]: return
            _fired[0] = True
            if self.gripper_timer_handle is not None:
                try: self.gripper_timer_handle.cancel()
                except: pass
                self.gripper_timer_handle = None
            self.gripper_en_progreso = False
            self.get_logger().info(f"[GRIPPER] '{cmd}' completado")
            try: on_done()
            except Exception as e:
                self.get_logger().error(f"Error gripper callback: {e}")
        self.gripper_timer_handle = self.create_timer(self.GRIPPER_WAIT_SEC, _fin)

    # ══════════════════════════════════════════════════════════════════════
    #  SPLASH / MOUSE / TECLAS
    # ══════════════════════════════════════════════════════════════════════

    def _splash(self):
        f = self.current_frame; f[:] = (30,30,30)
        cv2.putText(f, "PBVS Multicolor  [Offset + 2 fases]",
                    (30,70), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0,220,220), 2)
        cv2.putText(f, "Clic=añadir | k=iniciar | c=limpiar | z=quitar | Ctrl+C=salir",
                    (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,150), 1)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.permitir_arranque: return
        if self.last_raw_frame is None: return
        if self.estado in ("INICIALIZANDO","CONFIGURANDO_BRAZO",
                           "ESPERANDO_HOME","PROBANDO_GRIPPER_INICIAL",
                           "ERROR_GRIPPER_SIN_SUBSCRIBER"): return
        h,s,v = muestrear_hsv_parche(self.last_raw_frame, x, y)
        obj = ObjetivoColor(h,s,v)
        self.cola_objetivos.append(obj)
        self.get_logger().info(
            f"[CLIC #{len(self.cola_objetivos)}] HSV=({h},{s},{v}) → '{obj.etiqueta}'")

    def _procesar_tecla(self, key):
        if key in (-1, 255): return
        if key == ord('c'):
            if self.permitir_arranque: return
            self.cola_objetivos.clear()
        elif key == ord('z'):
            if self.permitir_arranque: return
            if self.cola_objetivos: self.cola_objetivos.pop()
        elif key == ord('k'):
            if self.permitir_arranque: return
            if not self.cola_objetivos: return
            if self.estado != "ESPERANDO_SELECCION": return
            self.get_logger().info(
                f"=== INICIANDO con {len(self.cola_objetivos)} objetivos ===")
            self.permitir_arranque = True
            self.idx_actual = 0
            self._cargar_objetivo(0)

    # ══════════════════════════════════════════════════════════════════════
    #  GESTIÓN DE OBJETIVOS
    # ══════════════════════════════════════════════════════════════════════

    def _cargar_objetivo(self, idx):
        if idx >= len(self.cola_objetivos):
            self.get_logger().info("=== COLA TERMINADA ===")
            self.cola_objetivos.clear()
            self.idx_actual = 0
            self.permitir_arranque = False
            self.objetivo_actual = None
            self.target_pixel = None; self.target_raw = None
            self.area_objeto = 0.0
            self.frames_perdidos = 0; self.frames_centrado = 0
            self.kf_x.reset(); self.kf_y.reset(); self._last_t = None
            self.pid_x.reset(); self.pid_y.reset()
            self.fase_pick = None
            self.ultimo_descenso_pick_mm = None
            self._descenso_fase = 0; self._descenso_restante_mm = 0.0
            self.estado = "ESPERANDO_SELECCION"
            return
        self.objetivo_actual = self.cola_objetivos[idx]
        self.idx_actual = idx
        self.target_pixel = None; self.target_raw = None
        self.area_objeto = 0.0
        self.ultimo_descenso_pick_mm = None
        self.frames_perdidos = 0; self.frames_centrado = 0
        self.kf_x.reset(); self.kf_y.reset(); self._last_t = None
        self.pid_x.reset(); self.pid_y.reset()
        self.fase_pick = None
        self._descenso_fase = 0; self._descenso_restante_mm = 0.0
        self.estado = "BUSCANDO_COLOR"
        self.get_logger().info(
            f"--- Objetivo {idx+1}/{len(self.cola_objetivos)}: "
            f"'{self.objetivo_actual.etiqueta}' ---")

    def esp32_callback(self, msg):
        self.distancia_esp32 = msg.data

    # ══════════════════════════════════════════════════════════════════════
    #  INICIALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════

    def iniciar_configuracion(self):
        if (not self.cli_set_mode.service_is_ready() or
                not self.cli_set_state.service_is_ready()): return
        self.timer_inicio.cancel()
        self._llamar_set_mode()

    def _llamar_set_mode(self):
        self.estado = "CONFIGURANDO_BRAZO"
        req = SetInt16.Request(); req.data = 0
        self.cli_set_mode.call_async(req).add_done_callback(self._set_mode_cb)

    def _set_mode_cb(self, future):
        try: self.get_logger().info(f"set_mode OK (ret={future.result().ret})")
        except Exception as e: self.get_logger().error(f"set_mode err: {e}")
        self._llamar_set_state()

    def _llamar_set_state(self):
        req = SetInt16.Request(); req.data = 0
        self.cli_set_state.call_async(req).add_done_callback(self._set_state_cb)

    def _set_state_cb(self, future):
        try: self.get_logger().info(f"set_state OK (ret={future.result().ret})")
        except Exception as e: self.get_logger().error(f"set_state err: {e}")
        self._ir_a_home_inicial()

    def _ir_a_home_inicial(self):
        self.estado = "ESPERANDO_HOME"
        self.movimiento_en_progreso = True
        req = MoveJoint.Request()
        req.angles = [math.radians(a) for a in self.HOME_JOINTS_DEG]
        req.speed = 0.4
        self.cli_joint.call_async(req).add_done_callback(self._home_ini_cb)

    def _home_ini_cb(self, future):
        self.movimiento_en_progreso = False
        try:
            if future.result().ret == 0:
                self._verificar_gripper_inicial()
            else:
                self.get_logger().error("HOME falló.")
        except Exception as e:
            self.get_logger().error(f"HOME err: {e}")

    def _verificar_gripper_inicial(self):
        self.estado = "PROBANDO_GRIPPER_INICIAL"
        if self.pub_gripper.get_subscription_count() == 0:
            self.get_logger().error("No hay subscriber en /lite6_gripper/command")
            self.estado = "ERROR_GRIPPER_SIN_SUBSCRIBER"
            return
        self._gripper_accion('open', self._gripper_ini_done)

    def _gripper_ini_done(self):
        self.get_logger().info("=== LISTO — selecciona colores y presiona 'k' ===")
        self.estado = "ESPERANDO_SELECCION"

    # ══════════════════════════════════════════════════════════════════════
    #  VISIÓN
    # ══════════════════════════════════════════════════════════════════════

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.last_raw_frame = frame.copy()
        est = self.estado
        permitir = est in ("BUSCANDO_COLOR", "SERVOING",
                           "RECENTRANDO_FASE2")

        if (self.objetivo_actual is not None
                and est in ("BUSCANDO_COLOR", "SERVOING",
                            "RECENTRANDO_FASE2",
                            "CENTRADO_FIJO", "CENTRADO_FIJO_FASE2",
                            "AGARRANDO")):
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = self.objetivo_actual.construir_mascara(hsv)
            self.mask_global = mask
            contornos, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detectado = False
            if contornos:
                max_c = max(contornos, key=cv2.contourArea)
                area  = cv2.contourArea(max_c)
                if area >= self.AREA_MIN_VALIDA:
                    M = cv2.moments(max_c)
                    if M["m00"] > 0:
                        detectado = True
                        cx_r = M["m10"]/M["m00"]
                        cy_r = M["m01"]/M["m00"]
                        if permitir:
                            t_now = self.get_clock().now().nanoseconds*1e-9
                            dt = 0.033 if self._last_t is None \
                                else max(1e-3, min(0.5, t_now - self._last_t))
                            self._last_t = t_now
                            sx = self.kf_x.update(cx_r, dt)
                            sy = self.kf_y.update(cy_r, dt)
                            self.target_raw   = (int(cx_r), int(cy_r))
                            self.target_pixel = (int(sx), int(sy))
                            self.area_objeto  = area
                            self.frames_perdidos = 0
                            if est == "BUSCANDO_COLOR":
                                self.get_logger().info(
                                    f"'{self.objetivo_actual.etiqueta}' detectado → SERVOING")
                                self.estado = "SERVOING"
                                self.frames_centrado = 0
                            elif est == "RECENTRANDO_FASE2":
                                # Ya está en servoing de fase 2, solo actualiza
                                pass

                        cv2.drawContours(frame, [max_c], -1, (0,255,255), 2)
                        cv2.circle(frame, (int(cx_r),int(cy_r)), 4, (150,150,150), -1)
                        if self.target_pixel:
                            cv2.circle(frame, self.target_pixel, 7, (0,0,255), -1)

            if not detectado and permitir:
                self.frames_perdidos += 1
                if self.frames_perdidos >= self.FRAMES_PERDIDOS_MAX:
                    if self.estado == "SERVOING":
                        self.get_logger().warn("Objeto perdido → BUSCANDO_COLOR")
                        self.estado = "BUSCANDO_COLOR"
                    elif self.estado == "RECENTRANDO_FASE2":
                        self.get_logger().warn("Objeto perdido en fase 2 → RECENTRANDO_FASE2 (sigue buscando)")
                    self.target_pixel = None; self.target_raw = None
                    self.area_objeto = 0.0; self.frames_centrado = 0
                    self.kf_x.reset(); self.kf_y.reset()
                    self.pid_x.reset(); self.pid_y.reset()
                    self._last_t = None

        self._dibujar_hud(frame)
        self.current_frame = frame

    # ══════════════════════════════════════════════════════════════════════
    #  CONTROL LOOP
    # ══════════════════════════════════════════════════════════════════════

    def control_loop(self):
        if self.estado == "AGARRANDO":
            self._ejecutar_secuencia_pick_place(); return
        if self.estado in ("FINALIZADO", "ESPERANDO_SELECCION"):
            return

        # ── CENTRADO_FIJO: primera vez → offset + descenso fase 1 ──────
        if self.estado == "CENTRADO_FIJO":
            self._ejecutar_offset_y_descenso_fase1(); return

        # ── CENTRADO_FIJO_FASE2: segunda vez → offset + descenso fase 2 ─
        if self.estado == "CENTRADO_FIJO_FASE2":
            self._ejecutar_offset_y_descenso_fase2(); return

        # ── SERVOING (fase 1, desde arriba) ────────────────────────────
        if self.estado == "SERVOING":
            self._servoing_tick("CENTRADO_FIJO"); return

        # ── RECENTRANDO_FASE2 (servoing desde la mitad) ────────────────
        if self.estado == "RECENTRANDO_FASE2":
            self._servoing_tick("CENTRADO_FIJO_FASE2"); return

    def _servoing_tick(self, estado_centrado_destino):
        """Lógica de servoing compartida entre fase 1 y fase 2."""
        if self.movimiento_en_progreso or self.gripper_en_progreso:
            return
        if self.target_pixel is None:
            return

        ex = self.target_pixel[0] - self.CENTRO_CAMARA[0]
        ey = self.target_pixel[1] - self.CENTRO_CAMARA[1]

        if abs(ex) <= self.TOLERANCIA_XY and abs(ey) <= self.TOLERANCIA_XY:
            self.frames_centrado += 1
            self.get_logger().info(
                f"[CENTRADO] {self.frames_centrado}/{self.FRAMES_CENTRADO_MIN} "
                f"ex={ex:+.0f} ey={ey:+.0f}")
            if self.frames_centrado >= self.FRAMES_CENTRADO_MIN:
                self.get_logger().info(
                    f"Centrado confirmado → {estado_centrado_destino}")
                self.pid_x.reset(); self.pid_y.reset()
                self.estado = estado_centrado_destino
            return

        self.frames_centrado = 0
        ex_eff = 0.0 if abs(ex) < self.DEADZONE_PX else ex
        ey_eff = 0.0 if abs(ey) < self.DEADZONE_PX else ey
        if ex_eff == 0.0 and ey_eff == 0.0: return
        if self.SWAP_AXES: ex_eff, ey_eff = ey_eff, ex_eff

        dt = 0.10
        out_x = self.SIGN_X * self.pid_x.update(ey_eff, dt)
        out_y = self.SIGN_Y * self.pid_y.update(ex_eff, dt)
        step_x = float(np.clip(out_x, -self.STEP_MAX_XY, self.STEP_MAX_XY))
        step_y = float(np.clip(out_y, -self.STEP_MAX_XY, self.STEP_MAX_XY))
        self._pid_p_disp = self.pid_x.Kp * ey_eff
        self._pid_i_disp = self.pid_x.Ki * self.pid_x._integral
        self._pid_d_disp = self.pid_x.Kd * self.pid_x._d_filtered
        self.get_logger().info(
            f"[PID] ex={ex:+.0f} ey={ey:+.0f} dx={step_x:+.2f} dy={step_y:+.2f}")
        self._mover_cartesiano_rel(step_x, step_y, 0.0, self.VEL_XY)

    # ══════════════════════════════════════════════════════════════════════
    #  DESCENSO FASE 1: offset + bajar 50%
    # ══════════════════════════════════════════════════════════════════════

    def _ejecutar_offset_y_descenso_fase1(self):
        if self.movimiento_en_progreso: return
        if self.distancia_esp32 is None: return

        d = float(self.distancia_esp32)
        if d <= 0.0 or d > 300.0:
            d = self.FALLBACK_DESCENSO

        d_fase1 = d * self.DESCENSO_FRACCION_1
        d_fase2 = d - d_fase1
        self._descenso_restante_mm = d_fase2
        self._descenso_fase = 1
        self.ultimo_descenso_pick_mm = d

        self.get_logger().info(
            f"[FASE 1] Offset cam→grip X={self.OFFSET_CAMARA_X:+.1f} "
            f"Y={self.OFFSET_CAMARA_Y:+.1f} mm → bajar {d_fase1:.1f} mm "
            f"(quedan {d_fase2:.1f} mm para fase 2)")

        # Primero aplicar offset XY, luego bajar
        self.estado = "AGARRANDO"
        self.fase_pick = "APLICANDO_OFFSET_F1"
        self._mover_cartesiano_rel_con_callback(
            self.OFFSET_CAMARA_X, self.OFFSET_CAMARA_Y, 0.0,
            self.VEL_XY, lambda: self._offset_f1_done(d_fase1))

    def _offset_f1_done(self, d_fase1):
        self.get_logger().info(f"[FASE 1] Offset aplicado → bajando {d_fase1:.1f} mm")
        self.fase_pick = "ESPERANDO_BAJADA_F1"
        self._mover_cartesiano_rel_con_callback(
            0.0, 0.0, -d_fase1, self.VEL_Z, self._bajada_f1_completa)

    def _bajada_f1_completa(self):
        self.get_logger().info(
            "[FASE 1] Bajada completa → re-centrando visualmente (fase 2)…")
        # Volver a servoing para re-centrar desde más cerca
        self.frames_centrado = 0
        self.frames_perdidos = 0
        self.kf_x.reset(); self.kf_y.reset(); self._last_t = None
        self.pid_x.reset(); self.pid_y.reset()
        self.target_pixel = None; self.target_raw = None
        self._descenso_fase = 1
        self.fase_pick = None
        self.estado = "RECENTRANDO_FASE2"

    # ══════════════════════════════════════════════════════════════════════
    #  DESCENSO FASE 2: offset + bajar restante + agarrar
    # ══════════════════════════════════════════════════════════════════════

    def _ejecutar_offset_y_descenso_fase2(self):
        if self.movimiento_en_progreso: return

        d_fase2 = self._descenso_restante_mm
        self._descenso_fase = 2

        self.get_logger().info(
            f"[FASE 2] Offset cam→grip X={self.OFFSET_CAMARA_X:+.1f} "
            f"Y={self.OFFSET_CAMARA_Y:+.1f} mm → bajar {d_fase2:.1f} mm → agarrar")

        self.estado = "AGARRANDO"
        self.fase_pick = "APLICANDO_OFFSET_F2"
        self._mover_cartesiano_rel_con_callback(
            self.OFFSET_CAMARA_X, self.OFFSET_CAMARA_Y, 0.0,
            self.VEL_XY, lambda: self._offset_f2_done(d_fase2))

    def _offset_f2_done(self, d_fase2):
        self.get_logger().info(f"[FASE 2] Offset aplicado → bajando {d_fase2:.1f} mm")
        self.fase_pick = "ESPERANDO_BAJADA_F2"
        self._mover_cartesiano_rel_con_callback(
            0.0, 0.0, -d_fase2, self.VEL_Z, self._bajada_f2_completa)

    def _bajada_f2_completa(self):
        self.get_logger().info(
            f"[FASE 2] Bajada terminada → esperando "
            f"{self.ESPERA_ANTES_CERRAR_SEC:.0f} s antes de cerrar.")
        self.fase_pick = "ESPERANDO_ESTABILIZACION"
        _timer = [None]
        def _on_timeout():
            _timer[0].cancel()
            self.get_logger().info("[ESPERA] Tiempo cumplido → cerrando gripper.")
            self.fase_pick = "CERRANDO"
        _timer[0] = self.create_timer(self.ESPERA_ANTES_CERRAR_SEC, _on_timeout)

    # ══════════════════════════════════════════════════════════════════════
    #  SECUENCIA PICK & PLACE
    # ══════════════════════════════════════════════════════════════════════

    def _ejecutar_secuencia_pick_place(self):
        if self.gripper_en_progreso or self.movimiento_en_progreso: return

        ESPERAS = {
            "APLICANDO_OFFSET_F1", "ESPERANDO_BAJADA_F1",
            "APLICANDO_OFFSET_F2", "ESPERANDO_BAJADA_F2",
            "ESPERANDO_ESTABILIZACION", "ESPERANDO_CERRADO",
            "ESPERANDO_LEVANTAMIENTO", "ESPERANDO_VIAJE_DROP",
            "ESPERANDO_BAJADA_DROP", "ESPERANDO_SOLTADO",
            "ESPERANDO_HOME",
        }
        if self.fase_pick in ESPERAS: return

        if self.fase_pick == "CERRANDO":
            self.get_logger().info("Cerrando gripper…")
            self.fase_pick = "ESPERANDO_CERRADO"
            self._gripper_accion('close', self._grip_cerr_done)

        elif self.fase_pick == "LEVANTANDO":
            self.get_logger().info(f"Levantando {self.ALTURA_LEVANTE} mm…")
            self.fase_pick = "ESPERANDO_LEVANTAMIENTO"
            self._mover_cartesiano_rel_con_callback(
                0.0, 0.0, self.ALTURA_LEVANTE, self.VEL_Z, self._levant_done)

        elif self.fase_pick == "VIAJANDO_DROP":
            etq = self.objetivo_actual.etiqueta
            pose = self.DROP_POSES.get(etq, self.DROP_POSES["desconocido"])
            self.get_logger().info(f"Viajando a drop '{etq}'")
            self.fase_pick = "ESPERANDO_VIAJE_DROP"
            self._mover_cartesiano_abs_con_callback(
                pose, self.VEL_TRAVEL, self._viaje_drop_done)

        elif self.fase_pick == "BAJANDO_DROP":
            d_base = self.ultimo_descenso_pick_mm or self.FALLBACK_DESCENSO
            d_drop = 0.5 * d_base
            self.get_logger().info(f"Bajando en drop {d_drop:.1f} mm")
            self.fase_pick = "ESPERANDO_BAJADA_DROP"
            self._mover_cartesiano_rel_con_callback(
                0.0, 0.0, -d_drop, self.VEL_Z, self._bajada_drop_done)

        elif self.fase_pick == "SOLTANDO":
            self.get_logger().info("Abriendo gripper…")
            self.fase_pick = "ESPERANDO_SOLTADO"
            self._gripper_accion('open', self._grip_solt_done)

        elif self.fase_pick == "VIAJANDO_HOME":
            self.get_logger().info("Regresando a HOME…")
            req = MoveJoint.Request()
            req.angles = [math.radians(a) for a in self.HOME_JOINTS_DEG]
            req.speed = 0.4
            self.movimiento_en_progreso = True
            self.fase_pick = "ESPERANDO_HOME"
            self.cli_joint.call_async(req).add_done_callback(self._home_pick_cb)

        elif self.fase_pick == "LISTO_SIGUIENTE":
            self.get_logger().info(f"Objetivo {self.idx_actual+1} completado.")
            self._cargar_objetivo(self.idx_actual + 1)

    def _grip_cerr_done(self):
        self.fase_pick = "LEVANTANDO"
    def _levant_done(self):
        self.fase_pick = "VIAJANDO_DROP"
    def _viaje_drop_done(self):
        self.fase_pick = "BAJANDO_DROP"
    def _bajada_drop_done(self):
        self.fase_pick = "SOLTANDO"
    def _grip_solt_done(self):
        self.fase_pick = "VIAJANDO_HOME"
    def _home_pick_cb(self, future):
        self.movimiento_en_progreso = False
        self.fase_pick = "LISTO_SIGUIENTE"

    # ══════════════════════════════════════════════════════════════════════
    #  MOVIMIENTOS CARTESIANOS
    # ══════════════════════════════════════════════════════════════════════

    def _mover_cartesiano_rel(self, dx, dy, dz, speed):
        if not self.cli_cartesian.service_is_ready(): return
        req = MoveCartesian.Request()
        req.pose = [dx, dy, dz, 0.0, 0.0, 0.0]
        req.speed = speed; req.relative = True
        self.movimiento_en_progreso = True
        self.cli_cartesian.call_async(req).add_done_callback(self._lib)

    def _mover_cartesiano_rel_con_callback(self, dx, dy, dz, speed, on_done):
        if not self.cli_cartesian.service_is_ready(): return
        req = MoveCartesian.Request()
        req.pose = [dx, dy, dz, 0.0, 0.0, 0.0]
        req.speed = speed; req.relative = True
        self.movimiento_en_progreso = True
        def _cb(f): self.movimiento_en_progreso = False; on_done()
        self.cli_cartesian.call_async(req).add_done_callback(_cb)

    def _mover_cartesiano_abs_con_callback(self, pose, speed, on_done):
        if not self.cli_cartesian.service_is_ready(): return
        req = MoveCartesian.Request()
        req.pose = list(pose); req.speed = speed; req.relative = False
        self.movimiento_en_progreso = True
        def _cb(f): self.movimiento_en_progreso = False; on_done()
        self.cli_cartesian.call_async(req).add_done_callback(_cb)

    def _lib(self, future):
        self.movimiento_en_progreso = False

    # ══════════════════════════════════════════════════════════════════════
    #  HUD
    # ══════════════════════════════════════════════════════════════════════

    def _dibujar_hud(self, frame):
        cx, cy = self.CENTRO_CAMARA
        cv2.drawMarker(frame, (cx,cy), (255,80,0), cv2.MARKER_CROSS, 40, 2)
        t = int(self.TOLERANCIA_XY); d = int(self.DEADZONE_PX)
        cv2.rectangle(frame, (cx-t,cy-t), (cx+t,cy+t), (0,200,0), 1)
        cv2.rectangle(frame, (cx-d,cy-d), (cx+d,cy+d), (200,200,0), 1)

        COL = {
            "INICIALIZANDO":(160,160,160), "CONFIGURANDO_BRAZO":(100,100,255),
            "ESPERANDO_HOME":(0,180,255), "PROBANDO_GRIPPER_INICIAL":(0,180,200),
            "ERROR_GRIPPER_SIN_SUBSCRIBER":(0,0,255),
            "ESPERANDO_SELECCION":(180,220,0), "BUSCANDO_COLOR":(0,220,180),
            "SERVOING":(0,255,0), "RECENTRANDO_FASE2":(0,255,180),
            "CENTRADO_FIJO":(255,200,0), "CENTRADO_FIJO_FASE2":(255,160,0),
            "AGARRANDO":(0,140,255), "FINALIZADO":(0,255,120),
        }
        c = COL.get(self.estado, (255,255,255))
        cv2.rectangle(frame, (0,0), (640,30), (20,20,20), -1)

        fase_txt = self.fase_pick or '-'
        if self.estado == "RECENTRANDO_FASE2":
            fase_txt = "re-centrado fino"
        cv2.putText(frame,
            f"Estado: {self.estado}  |  Fase: {fase_txt}",
            (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 2)

        # Cola
        xc = 470; yc = 50
        cv2.putText(frame, f"Cola ({len(self.cola_objetivos)}):",
                    (xc,yc-8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220,220,220), 1)
        for i, obj in enumerate(self.cola_objetivos):
            y = yc + i*26
            if y > 460: break
            act = (self.permitir_arranque and i == self.idx_actual
                   and self.estado not in ("ESPERANDO_SELECCION","FINALIZADO"))
            done = (self.permitir_arranque and i < self.idx_actual)
            cv2.rectangle(frame, (xc,y), (xc+22,y+20), obj.bgr_chip, -1)
            cv2.rectangle(frame, (xc,y), (xc+22,y+20),
                          (255,255,255) if act else (60,60,60), 2 if act else 1)
            tk = " V" if done else (" <" if act else "")
            cv2.putText(frame, f"{i+1}. {obj.etiqueta}{tk}",
                        (xc+30,y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (180,255,180) if done else (255,255,255) if act else (200,200,200), 1)

        if self.objetivo_actual and self.permitir_arranque:
            cv2.putText(frame,
                f"Buscando: {self.objetivo_actual.etiqueta} "
                f"({self.idx_actual+1}/{len(self.cola_objetivos)})",
                (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                self.objetivo_actual.bgr_chip, 2)

        if self.target_pixel:
            ex = self.target_pixel[0] - cx
            ey = self.target_pixel[1] - cy
            cv2.line(frame, self.target_pixel, (cx,cy), (0,255,255), 2)
            cv2.putText(frame, f"Ex:{ex:+.0f} Ey:{ey:+.0f}",
                        (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
            cv2.putText(frame,
                f"Centrado:{self.frames_centrado}/{self.FRAMES_CENTRADO_MIN}",
                (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,100), 1)

        cv2.putText(frame,
            f"PID P={self._pid_p_disp:+.3f} I={self._pid_i_disp:+.3f} "
            f"D={self._pid_d_disp:+.3f}",
            (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,180,50), 1)

        d_act = self.distancia_esp32 if self.distancia_esp32 is not None \
                else self.FALLBACK_DESCENSO
        fase_info = f"  fase={self._descenso_fase}" if self._descenso_fase > 0 else ""
        cv2.putText(frame,
            f"Descenso: {d_act:.0f} mm  "
            f"(50%+50%  offset X={self.OFFSET_CAMARA_X:+.0f} Y={self.OFFSET_CAMARA_Y:+.0f})"
            f"{fase_info}",
            (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)

        if self.gripper_en_progreso:
            cv2.putText(frame, "Gripper en movimiento...",
                        (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 2)

        if self.estado == "ESPERANDO_SELECCION":
            cv2.putText(frame,
                "Clic=añadir | k=iniciar | c=limpiar | z=quitar | Ctrl+C=salir",
                (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,255,180), 1)

    # ══════════════════════════════════════════════════════════════════════
    #  DISPLAY
    # ══════════════════════════════════════════════════════════════════════

    def display_loop(self):
        cv2.imshow("Control PBVS Multicolor", self.current_frame)
        key = cv2.waitKey(1) & 0xFF
        self._procesar_tecla(key)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = PBVSMulticolorTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C — cerrando.")
    finally:
        cv2.destroyAllWindows()
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
