import rclpy
from rclpy.node import Node
from xarm_msgs.srv import MoveCartesian
import time

class WaypointNode(Node):
    def __init__(self):
        super().__init__('lite6_waypoint_node')
        
        # --- CAMBIA ESTO POR EL NOMBRE DE TU SERVICIO ---
        service_name = '/ufactory/set_position' 
        # ------------------------------------------------
        
        self.cli = self.create_client(MoveCartesian, service_name)
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Esperando al servicio {service_name}...')
            
        self.req = MoveCartesian.Request()

    def send_waypoint(self, pose):
        # pose es una lista: [x, y, z, roll, pitch, yaw]
        # x, y, z están en milímetros (mm)
        # roll, pitch, yaw están en radianes
        self.req.pose = pose
        self.req.speed = 100.0  # Velocidad en mm/s
        self.req.acc = 500.0    # Aceleración en mm/s^2
        self.req.mvtime = 0.0

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()

    # Definimos dos waypoints seguros frente al robot
    # [X, Y, Z, Roll, Pitch, Yaw]
    waypoint_1 = [250.0, 50.0, 200.0, 3.14, 0.0, 0.0]
    waypoint_2 = [250.0, -50.0, 200.0, 3.14, 0.0, 0.0]

    node.get_logger().info("Moviendo al Waypoint 1...")
    node.send_waypoint(waypoint_1)
    time.sleep(1) # Pausa de 1 segundo

    node.get_logger().info("Moviendo al Waypoint 2...")
    node.send_waypoint(waypoint_2)

    node.get_logger().info("¡Movimientos completados!")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
