import pybullet as p
import numpy as np

class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p_id.createVisualShape(
            self.p_id.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=0)

        self.marker_id = self.p_id.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                self.p_id.addUserDebugText(text, position + radius)
            )

        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        self.p_id.removeBody(self.marker_id, physicsClientId=0)
        for debug_item_id in self.debug_item_ids:
            self.p_id.removeUserDebugItem(debug_item_id)