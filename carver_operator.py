import bpy
import sys

from bpy.props import (
	BoolProperty,
	IntProperty,
	PointerProperty,
	StringProperty,
	)
from mathutils import (
	Vector,
	)

from .carver_profils import (
	Profils
	)

from .carver_utils import (
	duplicateObject,
	UndoListUpdate,
	CreateCutLine,
	createMeshFromData,
	SelectObject,
	Selection_Save_Restore,
	Selection_Save,
	Selection_Restore,
	update_grid,
	objDiagonal,
	Undo,
	UndoAdd,
	Pick,
	rot_axis_quat,
	MoveCursor,
	Picking,
	CreateCutSquare,
	CreateCutCircle,
	CreateCutLine,
	CreatePrimitive,
	boolean_operation,
	update_bevel,
	CreateBevel,
	Rebool,
	)

from .carver_draw import draw_callback_px

# Modal Operator
class CARVER_OT_operator(bpy.types.Operator):
	bl_idname = "carver.operator"
	bl_label = "Carver"
	bl_description = "Cut or create Meshes in Object mode"
	bl_options = {'REGISTER', 'UNDO'}

	def __init__(self):
		context = bpy.context
		# Cut Type
		self.rectangle = 0
		self.line = 1
		self.circle = 2

		# Boolean operation
		self.difference = 0
		self.union = 1

		self.CurrentSelection = context.selected_objects.copy()
		self.CurrentActive = context.active_object
		self.SavSel = context.selected_objects.copy()
		self.Sav_ac = None

		args = (self, context)

		self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')

		self.mouse_path = [(0, 0), (0, 0)]

		self.shift = False
		self.ctrl = False
		self.alt = False

		self.DontApply = context.scene.mesh_carver.DontApply
		self.Auto_BevelUpdate = True

		# Cut type (Rectangle, Circle, Line)
		self.CutMode = False
		self.ObjectMode = False
		self.ProfileMode = False

		self.CutType = 0
		self.BoolOps = self.difference

		# Circle variables
		self.stepAngle = [2, 4, 5, 6, 9, 10, 15, 20, 30, 40, 45, 60, 72, 90]
		self.step = 4
		self.step_rotation = 0

		# Primitives Position
		self.xpos = 0
		self.ypos = 0
		self.InitPosition = False

		# Line Increment
		self.Increment = 15
		# Close polygonal shape
		self.Closed = False

		# Depth Cursor
		self.snapCursor = context.scene.mesh_carver.DepthCursor

		# Help
		self.AskHelp = False

		# Working object
		self.OpsObj = context.active_object

		# Create mode
		self.CreateMode = False
		self.ExclusiveCreateMode = False
		if len(context.selected_objects) == 0:
			self.ExclusiveCreateMode = True
			self.CreateMode = True

		# Rebool forced (cut line)
		self.ForceRebool = False

		self.ViewVector = Vector()
		self.CurrentObj = None

		# Brush
		self.BrushSolidify = False
		self.WidthSolidify = False
		self.CarveDepth = False
		self.BrushDepth = False
		self.BrushDepthOffset = 0.0

		self.ObjectScale = False

		self.CircleListRaw = []
		self.CLR_C = []
		self.CurLoc = Vector((0.0, 0.0, 0.0))
		self.SavCurLoc = Vector((0.0, 0.0, 0.0))
		self.CRadius = 1.0
		CreatePrimitive(self, 10.0, 1.0)
		self.VertsList = []
		self.FacesList = []

		self.am = -1, -1
		self.SavMousePos = None
		self.xSavMouse = 0

		self.ascale = 0
		self.aRotZ = 0
		self.nRotZ = 0
		self.quat_rot_axis = None
		self.quat_rot = None

		self.RandomRotation = context.scene.mesh_carver.ORandom

		self.ShowCursor = True

		self.Instantiate = context.scene.mesh_carver.OInstanciate

		self.ProfileBrush = None
		self.ObjectBrush = None
		self.InitBrushPosition = None
		self.InitBrushScale = None
		self.InitBrushQRotation = None
		self.InitBrushERotation = None
		self.InitBrushARotation = None

		self.ObjectBrush_DT = "WIRE"
		self.XRay = False

		# Grid mesh
		self.nbcol = 1
		self.nbrow = 1
		self.gapx = 0
		self.gapy = 0
		self.scale_x = 1
		self.scale_y = 1

		self.GridScaleX = False
		self.GridScaleY = False

	@classmethod
	def poll(cls, context):
		ob = None
		if len(context.selected_objects) > 0:
			ob = context.selected_objects[0]
		# Test if selected object or none (for create mode)
		return (
			(ob and ob.type == 'MESH' and context.mode == 'OBJECT') or
			(context.mode == 'OBJECT' and ob is None) or
			(context.mode == 'EDIT_MESH'))

	def modal(self, context, event):
		PI = 3.14156
		region_types = {'WINDOW', 'UI'}
		win = context.window


		for area in win.screen.areas:
			if area.type in ('VIEW_3D'):
				for region in area.regions:
					if not region_types or region.type in region_types:
						region.tag_redraw()

		if event.type in {
				'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
				'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_6',
				'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_5'}:
			return {'PASS_THROUGH'}
		try:
			# [Shift]
			self.shift = True if event.shift else False

			# [Ctrl]
			self.ctrl = True if event.ctrl else False

			# [Alt]
			self.alt = False
			if event.alt:
				if self.InitPosition is False:
					# Initialise position variable for start position
					self.xpos = 0
					self.ypos = 0
					self.last_mouse_region_x = event.mouse_region_x
					self.last_mouse_region_y = event.mouse_region_y
					self.InitPosition = True
				self.alt = True
			# [Alt] release
			if self.InitPosition and self.alt is False:
				# Update coordinates
				for i in range(0, len(self.mouse_path)):
					l = list(self.mouse_path[i])
					l[0] += self.xpos
					l[1] += self.ypos
					self.mouse_path[i] = tuple(l)

				self.xpos = self.ypos = 0
				self.InitPosition = False

			# Mode change (cut type)
			if event.type == 'SPACE' and event.value == 'PRESS':
				if self.ObjectMode or self.ProfileMode:
					# If grid, remove double with intersect meshes
					if ((self.nbcol + self.nbrow) > 3):
						# Go in edit mode mode
						bpy.ops.object.mode_set(mode='EDIT')
						# Remove duplicate vertices
						bpy.ops.mesh.remove_doubles()
						# Return in object mode
						bpy.ops.object.mode_set(mode='OBJECT')

					if self.alt:
						# Save selected objects
						self.SavSel = context.selected_objects.copy()
						if len(context.selected_objects) > 0:
							bpy.ops.object.select_all(action='TOGGLE')

						if self.ObjectMode:
							SelectObject(self, self.ObjectBrush)
						else:
							SelectObject(self, self.ProfileBrush)
						duplicateObject(self)
					else:
						# Brush Cut
						self.Cut()
						# Save selected objects
						if self.ObjectMode:
							if len(self.ObjectBrush.children) > 0:
								self.SavSel = context.selected_objects.copy()
								if len(context.selected_objects) > 0:
									bpy.ops.object.select_all(action='TOGGLE')

								if self.ObjectMode:
									SelectObject(self, self.ObjectBrush)
								else:
									SelectObject(self, self.ProfileBrush)
								duplicateObject(self)

						UndoListUpdate(self)

					# Save cursor position
					self.SavMousePos = self.CurLoc
				else:
					if self.CutMode is False:
						# Cut Mode
						self.CutType += 1
						if self.CutType > 2:
							self.CutType = 0
					else:
						if self.CutType == self.line:
							# Cuts creation
							CreateCutLine(self, context)
							if self.CreateMode:
								# Object creation
								self.CreateGeometry()
								bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
								# Cursor Snap
								context.scene.mesh_carver.DepthCursor = self.snapCursor
								# Object Instantiate
								context.scene.mesh_carver.OInstanciate = self.Instantiate
								# Random rotation
								context.scene.mesh_carver.ORandom = self.RandomRotation

								return {'FINISHED'}
							else:
								self.Cut()
								UndoListUpdate(self)

			# Object creation
			if event.type == self.carver_prefs.Key_Create and event.value == 'PRESS':
				if self.ExclusiveCreateMode is False:
					self.CreateMode = not self.CreateMode

			# Auto Bevel Update
			if event.type == self.carver_prefs.Key_Update and event.value == 'PRESS':
				self.Auto_BevelUpdate = not self.Auto_BevelUpdate

			# Boolean operation type
			if event.type == self.carver_prefs.Key_Bool and event.value == 'PRESS':
				if (self.ProfileMode is True) or (self.ObjectMode is True):
					if self.BoolOps == self.difference:
						self.BoolOps = self.union
					else:
						self.BoolOps = self.difference

			# Brush Mode
			if event.type == self.carver_prefs.Key_Brush and event.value == 'PRESS':
				self.DontApply = False
				if (self.ProfileMode is False) and (self.ObjectMode is False):
					self.ProfileMode = True
				else:
					self.ProfileMode = False
					if self.ObjectBrush is not None:
						if self.ObjectMode is False:
							self.ObjectMode = True
							self.BrushSolidify = False
							self.CList = self.OB_List

							Selection_Save_Restore(self)
							context.scene.mesh_carver.nProfile = self.nProfil
						else:
							self.ObjectMode = False
					else:
						self.BrushSolidify = False
						Selection_Save_Restore(self)

				if self.ProfileMode:
					createMeshFromData(self)
					self.ProfileBrush = bpy.data.objects["CT_Profil"]
					Selection_Save(self)
					self.BrushSolidify = True

					bpy.ops.object.select_all(action='TOGGLE')
					self.ProfileBrush.select_set(True)
					context.view_layer.objects.active = self.ProfileBrush
					# Set xRay
					self.ProfileBrush.show_in_front = True

					bpy.ops.object.modifier_add(type='SOLIDIFY')
					context.object.modifiers["Solidify"].name = "CT_SOLIDIFY"
					context.object.modifiers["CT_SOLIDIFY"].thickness = 0.1

					Selection_Restore(self)

					self.CList = self.CurrentSelection
				else:
					if self.ObjectBrush is not None:
						if self.ObjectMode is False:
							if self.ObjectBrush is not None:
								self.ObjectBrush.location = self.InitBrushPosition
								self.ObjectBrush.scale = self.InitBrushScale
								self.ObjectBrush.rotation_quaternion = self.InitBrushQRotation
								self.ObjectBrush.rotation_euler = self.InitBrushERotation
								self.ObjectBrush.display_type = self.ObjectBrush_DT
								self.ObjectBrush.show_in_front = self.XRay

								# Remove solidify modifier
								Selection_Save(self)
								self.BrushSolidify = False

								bpy.ops.object.select_all(action='TOGGLE')
								self.ObjectBrush.select_set(True)
								context.view_layer.objects.active = self.ObjectBrush

								bpy.ops.object.modifier_remove(modifier="CT_SOLIDIFY")

								Selection_Restore(self)
						else:
							if self.Solidify_Active_Start:
								Selection_Save(self)
								self.BrushSolidify = True
								self.SolidifyPossible = True
								bpy.ops.object.select_all(action='TOGGLE')
								self.ObjectBrush.select_set(True)
								context.view_layer.objects.active = self.ObjectBrush
								# Set xRay
								self.ObjectBrush.show_in_front = True
								bpy.ops.object.modifier_add(type='SOLIDIFY')
								context.object.modifiers["Solidify"].name = "CT_SOLIDIFY"
								context.object.modifiers["CT_SOLIDIFY"].thickness = 0.1
								Selection_Restore(self)

			# Help display
			if event.type == self.carver_prefs.Key_Help and event.value == 'PRESS':
				self.AskHelp = not self.AskHelp

			# Instantiate object
			if event.type == self.carver_prefs.Key_Instant and event.value == 'PRESS':
				self.Instantiate = not self.Instantiate

			# Close polygonal shape
			if event.type == self.carver_prefs.Key_Close and event.value == 'PRESS':
				if self.CreateMode:
					self.Closed = not self.Closed

			if event.type == self.carver_prefs.Key_Apply and event.value == 'PRESS':
				self.DontApply = not self.DontApply

			# Scale object
			if event.type == self.carver_prefs.Key_Scale and event.value == 'PRESS':
				if self.ObjectScale is False:
					self.am = event.mouse_region_x, event.mouse_region_y
				self.ObjectScale = True

			# Grid : Add column
			if event.type == 'UP_ARROW' and event.value == 'PRESS':
				self.nbcol += 1
				update_grid(self, context)

			# Grid : Add row
			elif event.type == 'RIGHT_ARROW' and event.value == 'PRESS':
				self.nbrow += 1
				update_grid(self, context)

			# Grid : Delete column
			elif event.type == 'DOWN_ARROW' and event.value == 'PRESS':
				self.nbcol -= 1
				update_grid(self, context)

			# Grid : Delete row
			elif event.type == 'LEFT_ARROW' and event.value == 'PRESS':
				self.nbrow -= 1
				update_grid(self, context)

			# Grid : Scale gap between columns
			if event.type == self.carver_prefs.Key_Gapy and event.value == 'PRESS':
				if self.GridScaleX is False:
					self.am = event.mouse_region_x, event.mouse_region_y
				self.GridScaleX = True

			# Grid : Scale gap between rows
			if event.type == self.carver_prefs.Key_Gapx and event.value == 'PRESS':
				if self.GridScaleY is False:
					self.am = event.mouse_region_x, event.mouse_region_y
				self.GridScaleY = True

			# Cursor depth or solidify pattern
			if event.type == self.carver_prefs.Key_Depth and event.value == 'PRESS':
				if (self.ObjectMode is False) and (self.ProfileMode is False):
					self.snapCursor = not self.snapCursor
				else:
					# Solidify

					if (self.ObjectMode or self.ProfileMode) and (self.SolidifyPossible):
						solidify = True

						if self.ObjectMode:
							z = self.ObjectBrush.data.vertices[0].co.z
							ErrorMarge = 0.01
							for v in self.ObjectBrush.data.vertices:
								if abs(v.co.z - z) > ErrorMarge:
									solidify = False
									self.CarveDepth = True
									self.am = event.mouse_region_x, event.mouse_region_y
									break

						if solidify:
							if self.ObjectMode:
								for mb in self.ObjectBrush.modifiers:
									if mb.type == 'SOLIDIFY':
										AlreadySoldify = True
							else:
								for mb in self.ProfileBrush.modifiers:
									if mb.type == 'SOLIDIFY':
										AlreadySoldify = True

							if AlreadySoldify is False:
								Selection_Save(self)
								self.BrushSolidify = True

								bpy.ops.object.select_all(action='TOGGLE')
								if self.ObjectMode:
									self.ObjectBrush.select_set(True)
									context.view_layer.objects.active = self.ObjectBrush
									# Active le xray
									self.ObjectBrush.show_in_front = True
								else:
									self.ProfileBrush.select_set(True)
									context.view_layer.objects.active = self.ProfileBrush
									# Active le xray
									self.ProfileBrush.show_in_front = True

								bpy.ops.object.modifier_add(type='SOLIDIFY')
								context.object.modifiers["Solidify"].name = "CT_SOLIDIFY"

								context.object.modifiers["CT_SOLIDIFY"].thickness = 0.1

								Selection_Restore(self)

							self.WidthSolidify = not self.WidthSolidify
							self.am = event.mouse_region_x, event.mouse_region_y

			if event.type == self.carver_prefs.Key_BrushDepth and event.value == 'PRESS':
				if self.ObjectMode:
					self.CarveDepth = False

					self.BrushDepth = True
					self.am = event.mouse_region_x, event.mouse_region_y

			# Random rotation
			if event.type == 'R' and event.value == 'PRESS':
				self.RandomRotation = not self.RandomRotation

			# Undo
			if event.type == 'Z' and event.value == 'PRESS':
				if self.ctrl:
					if (self.CutType == self.line) and (self.CutMode):
						if len(self.mouse_path) > 1:
							self.mouse_path[len(self.mouse_path) - 1:] = []
					else:
						Undo(self)

			# Mouse move
			if event.type == 'MOUSEMOVE':
				# print("self.ObjectMode: ", self.ObjectMode)
				# print("self.ProfileMode: ", self.ProfileMode)
				# print("self.CutMode: ", self.CutMode)
				if self.ObjectMode or self.ProfileMode:
					fac = 50.0
					if self.shift:
						fac = 500.0
					if self.WidthSolidify:
						if self.ObjectMode:
							bpy.data.objects[self.ObjectBrush.name].modifiers[
								"CT_SOLIDIFY"].thickness += (event.mouse_region_x - self.am[0]) / fac
						elif self.ProfileMode:
							bpy.data.objects[self.ProfileBrush.name].modifiers[
								"CT_SOLIDIFY"].thickness += (event.mouse_region_x - self.am[0]) / fac
						self.am = event.mouse_region_x, event.mouse_region_y
					elif self.CarveDepth:
						for v in self.ObjectBrush.data.vertices:
							v.co.z += (event.mouse_region_x - self.am[0]) / fac
						self.am = event.mouse_region_x, event.mouse_region_y
					elif self.BrushDepth:
						self.BrushDepthOffset += (event.mouse_region_x - self.am[0]) / fac
						self.am = event.mouse_region_x, event.mouse_region_y
					else:
						if (self.GridScaleX):
							self.gapx += (event.mouse_region_x - self.am[0]) / 50
							self.am = event.mouse_region_x, event.mouse_region_y
							update_grid(self, context)
							return {'RUNNING_MODAL'}

						elif (self.GridScaleY):
							self.gapy += (event.mouse_region_x - self.am[0]) / 50
							self.am = event.mouse_region_x, event.mouse_region_y
							update_grid(self, context)
							return {'RUNNING_MODAL'}

						elif self.ObjectScale:
							self.ascale = -(event.mouse_region_x - self.am[0])
							self.am = event.mouse_region_x, event.mouse_region_y

							if self.ObjectMode:
								self.ObjectBrush.scale.x -= float(self.ascale) / 150.0
								if self.ObjectBrush.scale.x <= 0.0:
									self.ObjectBrush.scale.x = 0.0
								self.ObjectBrush.scale.y -= float(self.ascale) / 150.0
								if self.ObjectBrush.scale.y <= 0.0:
									self.ObjectBrush.scale.y = 0.0
								self.ObjectBrush.scale.z -= float(self.ascale) / 150.0
								if self.ObjectBrush.scale.z <= 0.0:
									self.ObjectBrush.scale.z = 0.0

							elif self.ProfileMode:
								if self.ProfileBrush is not None:
									self.ProfileBrush.scale.x -= float(self.ascale) / 150.0
									self.ProfileBrush.scale.y -= float(self.ascale) / 150.0
									self.ProfileBrush.scale.z -= float(self.ascale) / 150.0
						else:
							if self.LMB:
								if self.ctrl:
									self.aRotZ = - \
										((int((event.mouse_region_x - self.xSavMouse) / 10.0) * PI / 4.0) * 25.0)
								else:
									self.aRotZ -= event.mouse_region_x - self.am[0]
								self.ascale = 0.0

								self.am = event.mouse_region_x, event.mouse_region_y
							else:
								target_hit, target_normal, target_eul_rotation = Pick(context, event, self)
								if target_hit is not None:
									self.ShowCursor = True
									up_vector = Vector((0.0, 0.0, 1.0))
									quat_rot_axis = rot_axis_quat(up_vector, target_normal)
									self.quat_rot = target_eul_rotation @ quat_rot_axis
									MoveCursor(quat_rot_axis, target_hit, self)
									self.SavCurLoc = target_hit
									if self.ctrl:
										if self.SavMousePos is not None:
											xEcart = abs(self.SavMousePos.x - self.SavCurLoc.x)
											yEcart = abs(self.SavMousePos.y - self.SavCurLoc.y)
											zEcart = abs(self.SavMousePos.z - self.SavCurLoc.z)
											if (xEcart > yEcart) and (xEcart > zEcart):
												self.CurLoc = Vector(
													(target_hit.x, self.SavMousePos.y, self.SavMousePos.z))
											if (yEcart > xEcart) and (yEcart > zEcart):
												self.CurLoc = Vector(
													(self.SavMousePos.x, target_hit.y, self.SavMousePos.z))
											if (zEcart > xEcart) and (zEcart > yEcart):
												self.CurLoc = Vector(
													(self.SavMousePos.x, self.SavMousePos.y, target_hit.z))
											else:
												self.CurLoc = target_hit
									else:
										self.CurLoc = target_hit
				else:
					if self.CutMode:

						if self.alt is False:
							if self.CutMode:
								if self.ctrl and (self.CutType == self.line):
									# Incremental mode
									coord = list(self.mouse_path[len(self.mouse_path) - 1])
									coord[0] = int(
												self.mouse_path[len(self.mouse_path) - 2][0] +
												int((event.mouse_region_x -
													self.mouse_path[len(self.mouse_path) - 2][0]
													) / self.Increment) * self.Increment
												)
									coord[1] = int(
												self.mouse_path[len(self.mouse_path) - 2][1] +
												int((event.mouse_region_y -
													self.mouse_path[len(self.mouse_path) - 2][1]
													) / self.Increment) * self.Increment
												)
									self.mouse_path[len(self.mouse_path) - 1] = tuple(coord)
								else:
									if len(self.mouse_path) > 0:
										self.mouse_path[len(self.mouse_path) -
														1] = (event.mouse_region_x, event.mouse_region_y)
						else:
							# [ALT] press, update position
							self.xpos += (event.mouse_region_x - self.last_mouse_region_x)
							self.ypos += (event.mouse_region_y - self.last_mouse_region_y)

							self.last_mouse_region_x = event.mouse_region_x
							self.last_mouse_region_y = event.mouse_region_y

			elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
				if self.ObjectMode or self.ProfileMode:
					if self.LMB is False:
						target_hit, target_normal, target_eul_rotation  = Pick(context, event, self)
						if target_hit is not None:
							up_vector = Vector((0.0, 0.0, 1.0))
							self.quat_rot_axis = rot_axis_quat(up_vector, target_normal)
							self.quat_rot = target_eul_rotation @ self.quat_rot_axis
						self.am = event.mouse_region_x, event.mouse_region_y
						self.xSavMouse = event.mouse_region_x

						if self.ctrl:
							self.nRotZ = int((self.aRotZ / 25.0) / (PI / 4.0))
							self.aRotZ = self.nRotZ * (PI / 4.0) * 25.0

					self.LMB = True

			# LEFTMOUSE
			elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
				if self.ObjectMode or self.ProfileMode:
					# Rotation and scale
					self.LMB = False
					if self.ObjectScale is True:
						self.ObjectScale = False

					if self.GridScaleX is True:
						self.GridScaleX = False

					if self.GridScaleY is True:
						self.GridScaleY = False

					if self.WidthSolidify:
						self.WidthSolidify = False

					if self.CarveDepth is True:
						self.CarveDepth = False

					if self.BrushDepth is True:
						self.BrushDepth = False

				else:
					if self.CutMode is False:
						if self.ctrl:
							Picking(context, event)
						else:
							if self.CutType == self.line:
								if self.CutMode is False:
									self.mouse_path.clear()
									self.mouse_path.append((event.mouse_region_x, event.mouse_region_y))
									self.mouse_path.append((event.mouse_region_x, event.mouse_region_y))
							else:
								self.mouse_path[0] = (event.mouse_region_x, event.mouse_region_y)
								self.mouse_path[1] = (event.mouse_region_x, event.mouse_region_y)
							self.CutMode = True
					else:
						if self.CutType != self.line:
							# Cut creation
							if self.CutType == self.rectangle:
								CreateCutSquare(self, context)
							if self.CutType == self.circle:
								CreateCutCircle(self, context)
							if self.CutType == self.line:
								CreateCutLine(self, context)

							if self.CreateMode:
								# Object creation
								self.CreateGeometry()
								bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
								# Depth Cursor
								context.scene.mesh_carver.DepthCursor = self.snapCursor
								# Instantiate object
								context.scene.mesh_carver.OInstanciate = self.Instantiate
								# Random rotation
								context.scene.mesh_carver.ORandom = self.RandomRotation
								# Apply operation
								context.scene.mesh_carver.DontApply = self.DontApply

								# if Object mode, set intiale state
								if self.ObjectBrush is not None:
									self.ObjectBrush.location = self.InitBrushPosition
									self.ObjectBrush.scale = self.InitBrushScale
									self.ObjectBrush.rotation_quaternion = self.InitBrushQRotation
									self.ObjectBrush.rotation_euler = self.InitBrushERotation
									self.ObjectBrush.display_type = self.ObjectBrush_DT
									self.ObjectBrush.show_in_front = self.XRay

									# remove solidify
									Selection_Save(self)
									self.BrushSolidify = False

									bpy.ops.object.select_all(action='TOGGLE')
									self.ObjectBrush.select_set(True)
									context.view_layer.objects.active = self.ObjectBrush

									bpy.ops.object.modifier_remove(modifier="CT_SOLIDIFY")

									Selection_Restore(self)

									context.scene.mesh_carver.nProfile = self.nProfil

								return {'FINISHED'}
							else:
								self.Cut()
								UndoListUpdate(self)
						else:
							# Line
							self.mouse_path.append((event.mouse_region_x, event.mouse_region_y))

			# Change circle subdivisions
			elif (event.type == 'COMMA' and event.value == 'PRESS') or \
						(event.type == self.carver_prefs.Key_Subrem and event.value == 'PRESS'):
				if self.ProfileMode:
					self.nProfil += 1
					if self.nProfil >= self.MaxProfil:
						self.nProfil = 0
					createMeshFromData(self)
				# Circle rotation
				if self.CutType == self.circle:
					if self.ctrl:
						self.step_rotation += 1
					else:
						self.step += 1
						if self.step >= len(self.stepAngle):
							self.step = len(self.stepAngle) - 1
			elif (event.type == 'PERIOD' and event.value == 'PRESS') or \
						(event.type == self.carver_prefs.Key_Subadd and event.value == 'PRESS'):
				if self.ProfileMode:
					self.nProfil -= 1
					if self.nProfil < 0:
						self.nProfil = self.MaxProfil - 1
					createMeshFromData(self)
				if self.CutType == self.circle:
					if self.ctrl:
						self.step_rotation -= 1
					else:
						if self.step > 0:
							self.step -= 1
			# Quit
			elif event.type in {'RIGHTMOUSE', 'ESC'}:
				# Depth Cursor
				context.scene.mesh_carver.DepthCursor = self.snapCursor
				# Instantiate object
				context.scene.mesh_carver.OInstanciate = self.Instantiate
				# Random Rotation
				context.scene.mesh_carver.ORandom = self.RandomRotation
				# Apply boolean operation
				context.scene.mesh_carver.DontApply = self.DontApply

				# Reset Object
				if self.ObjectBrush is not None:
					self.ObjectBrush.location = self.InitBrushPosition
					self.ObjectBrush.scale = self.InitBrushScale
					self.ObjectBrush.rotation_quaternion = self.InitBrushQRotation
					self.ObjectBrush.rotation_euler = self.InitBrushERotation
					self.ObjectBrush.display_type = self.ObjectBrush_DT
					self.ObjectBrush.show_in_front = self.XRay

					# Remove solidify modifier
					Selection_Save(self)
					self.BrushSolidify = False

					bpy.ops.object.select_all(action='TOGGLE')
					self.ObjectBrush.select_set(True)
					context.view_layer.objects.active = self.ObjectBrush

					bpy.ops.object.modifier_remove(modifier="CT_SOLIDIFY")
					bpy.ops.object.select_all(action='TOGGLE')

					Selection_Restore(self)

				Selection_Save_Restore(self)
				context.view_layer.objects.active = self.CurrentActive
				context.scene.mesh_carver.nProfile = self.nProfil

				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

				# Remove Copy Object Brush
				if bpy.data.objects.get("CarverBrushCopy") is not None:
					brush = bpy.data.objects["CarverBrushCopy"]
					self.ObjectBrush.data = bpy.data.meshes[brush.data.name]
					bpy.ops.object.select_all(action='DESELECT')
					bpy.data.objects["CarverBrushCopy"].select_set(True)
					bpy.ops.object.delete()

				return {'FINISHED'}

			return {'RUNNING_MODAL'}

		except:
			print("\n[Carver MT ERROR]\n")
			import traceback
			traceback.print_exc()

			context.window.cursor_modal_set("DEFAULT")
			context.area.header_text_set(None)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

			self.report({'WARNING'},
						"Operation finished. Failure during Carving (Check the console for more info)")

			return {'FINISHED'}

	def cancel(self, context):
		# Note: used to prevent memory leaks on quitting Blender while the modal operator
		# is still running, gets called on return {"CANCELLED"}
		bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

	def invoke(self, context, event):
		if context.area.type != 'VIEW_3D':
			self.report({'WARNING'},
						"View3D not found or not currently active. Operation Cancelled")
			return {'CANCELLED'}

		if context.mode == 'EDIT_MESH':
			bpy.ops.object.mode_set(mode='OBJECT')

		# test if some other object types are selected that are not meshes
		test_selection = True
		self.carver_prefs = bpy.context.preferences.addons[__package__].preferences

		for obj in context.selected_objects:
			if obj.type != "MESH":
				test_selection = False
				break
		if not test_selection:
			self.report({'WARNING'},
						"Some selected objects are not of the Mesh type. Operation Cancelled")
			return {"CANCELLED"}

		# Get default patterns
		self.Profils = []
		for p in Profils:
			self.Profils.append((p[0], p[1], p[2], p[3]))

		for o in context.scene.objects:
			if not o.name.startswith(self.carver_prefs.ProfilePrefix):
				continue

			# In-scene profiles may have changed, remove them to refresh
			for m in bpy.data.meshes:
				if m.name.startswith(self.carver_prefs.ProfilePrefix):
					bpy.data.meshes.remove(m)

			vertices = []
			for v in o.data.vertices:
				vertices.append((v.co.x, v.co.y, v.co.z))

			faces = []
			for f in o.data.polygons:
				face = []
				for v in f.vertices:
					face.append(v)

				faces.append(face)

			self.Profils.append(
				(o.name,
				Vector((o.location.x, o.location.y, o.location.z)),
				vertices, faces)
			)

		self.nProfil = context.scene.mesh_carver.nProfile
		self.MaxProfil = len(self.Profils)

		# reset selected profile if last profile exceeds length of array
		if self.nProfil >= self.MaxProfil:
			self.nProfil = context.scene.mesh_carver.nProfile = 0


		if len(context.selected_objects) > 1:
			self.ObjectBrush = context.active_object

			# Copy the brush object
			ob = bpy.data.objects.new("CarverBrushCopy", context.object.data.copy())
			ob.location = self.ObjectBrush.location
			context.collection.objects.link(ob)
			context.scene.update()

			# Get default variables
			self.InitBrushPosition = self.ObjectBrush.location.copy()
			self.InitBrushScale = self.ObjectBrush.scale.copy()
			self.InitBrushQRotation = self.ObjectBrush.rotation_quaternion.copy()
			self.InitBrushERotation = self.ObjectBrush.rotation_euler.copy()
			self.ObjectBrush_DT = self.ObjectBrush.display_type
			self.XRay = self.ObjectBrush.show_in_front
			# Test if flat object
			z = self.ObjectBrush.data.vertices[0].co.z
			ErrorMarge = 0.01
			self.Solidify_Active_Start = True
			for v in self.ObjectBrush.data.vertices:
				if abs(v.co.z - z) > ErrorMarge:
					self.Solidify_Active_Start = False
					break
			self.SolidifyPossible = False

		self.CList = []
		self.OPList = []
		self.RList = []
		self.OB_List = []

		for ent in context.selected_objects:
			if ent != self.ObjectBrush:
				self.OB_List.append(ent)

		# Left button
		self.LMB = False

		# Undo Variables
		self.undo_index = 0
		self.undo_limit = context.preferences.edit.undo_steps
		self.undo_list = []

		# Boolean operations type
		self.BooleanType = 0

		self.UList = []
		self.UList_Index = -1
		self.UndoOps = []

		context.window_manager.modal_handler_add(self)
		return {'RUNNING_MODAL'}

	def CreateGeometry(self):
		context = bpy.context
		bLocalView = False

		for area in context.screen.areas:
			if area.type == 'VIEW_3D':
				if area.spaces[0].local_view is not None:
					bLocalView = True

		if bLocalView:
			bpy.ops.view3d.localview()

		if self.ExclusiveCreateMode:
			# Default width
			objBBDiagonal = 0.5
		else:
			ActiveObj = self.CurrentSelection[0]
			if ActiveObj is not None:
				# Object dimensions
				objBBDiagonal = objDiagonal(ActiveObj) / 4
		subdivisions = 2

		if len(context.selected_objects) > 0:
			bpy.ops.object.select_all(action='TOGGLE')

		context.view_layer.objects.active = self.CurrentObj

		bpy.data.objects[self.CurrentObj.name].select_set(True)
		bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

		bpy.ops.object.mode_set(mode='EDIT')
		bpy.ops.mesh.select_all(action='SELECT')
		bpy.ops.mesh.select_mode(type="EDGE")
		if self.snapCursor is False:
			bpy.ops.transform.translate(value=self.ViewVector * objBBDiagonal * subdivisions)
		bpy.ops.mesh.extrude_region_move(
			TRANSFORM_OT_translate={"value": -self.ViewVector * objBBDiagonal * subdivisions * 2})

		bpy.ops.mesh.select_all(action='SELECT')
		bpy.ops.mesh.normals_make_consistent()
		bpy.ops.object.mode_set(mode='OBJECT')

		saved_location_0 = context.scene.cursor.location.copy()
		bpy.ops.view3d.snap_cursor_to_active()
		saved_location = context.scene.cursor.location.copy()
		bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
		context.scene.cursor.location = saved_location
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
		context.scene.cursor.location = saved_location_0

		bpy.data.objects[self.CurrentObj.name].select_set(True)
		bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

		for o in self.SavSel:
			bpy.data.objects[o.name].select_set(True)

		if bLocalView:
			bpy.ops.view3d.localview()

		self.CutMode = False
		self.mouse_path.clear()
		self.mouse_path = [(0, 0), (0, 0)]

	def Cut(self):
		context = bpy.context

		# Local view ?
		bLocalView = False
		for area in context.screen.areas:
			if area.type == 'VIEW_3D':
				if area.spaces[0].local_view is not None:
					bLocalView = True

		if bLocalView:
			bpy.ops.view3d.localview()

		# Save cursor position
		CursorLocation = context.scene.cursor.location.copy()

		ActiveObjList = []
		if (self.ObjectMode is False) and (self.ProfileMode is False):
			objBBDiagonal = objDiagonal(self.CurrentSelection[0])
			subdivisions = 32
			if self.DontApply:
				subdivisions = 1

			# Get selected objects
			ActiveObjList = context.selected_objects.copy()

			bpy.ops.object.select_all(action='TOGGLE')

			context.view_layer.objects.active = self.CurrentObj

			bpy.data.objects[self.CurrentObj.name].select_set(True)
			bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.mesh.select_all(action='SELECT')
			bpy.ops.mesh.select_mode(type="EDGE")
			if (self.snapCursor is False) or (self.ForceRebool):
				bpy.ops.transform.translate(value=self.ViewVector * objBBDiagonal * subdivisions)
			bpy.ops.mesh.extrude_region_move(
				TRANSFORM_OT_translate={"value": -self.ViewVector * objBBDiagonal * subdivisions * 2})
			bpy.ops.mesh.select_all(action='SELECT')
			bpy.ops.mesh.normals_make_consistent()
			bpy.ops.object.mode_set(mode='OBJECT')
		else:
			# Create list
			if self.ObjectMode:
				for o in self.CurrentSelection:
					if o != self.ObjectBrush:
						ActiveObjList.append(o)
				self.CurrentObj = self.ObjectBrush
			else:
				ActiveObjList = self.CurrentSelection
				self.CurrentObj = self.ProfileBrush

		for o in self.CurrentSelection:
			UndoAdd(self, "MESH", o)

		# List objects create with rebool
		lastSelected = []

		for ActiveObj in ActiveObjList:
			context.scene.cursor.location = CursorLocation

			if len(context.selected_objects) > 0:
				bpy.ops.object.select_all(action='TOGGLE')

			# Test if initial object has bevel
			BevelAO = False
			for obj in ActiveObjList:
				for mb in obj.modifiers:
					if mb.type == 'BEVEL':
						BevelAO = True

			# Select cut object
			bpy.data.objects[self.CurrentObj.name].select_set(True)
			context.view_layer.objects.active = self.CurrentObj

			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.mesh.select_all(action='SELECT')
			bpy.ops.object.mode_set(mode='OBJECT')

			# Select object to cut
			bpy.data.objects[ActiveObj.name].select_set(True)
			context.view_layer.objects.active = ActiveObj

			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.mesh.select_all(action='DESELECT')
			bpy.ops.object.mode_set(mode='OBJECT')

			# Boolean operation
			if (self.shift is False) and (self.ForceRebool is False):
				if self.ObjectMode or self.ProfileMode:
					if self.BoolOps == self.union:
						boolean_operation(bool_type="UNION")
					else:
						boolean_operation(bool_type="DIFFERENCE")
				else:
					boolean_operation(bool_type="DIFFERENCE")

				# Apply booleans
				if self.DontApply is False:
					BMname = "CT_" + self.CurrentObj.name
					for mb in ActiveObj.modifiers:
						if (mb.type == 'BOOLEAN') and (mb.name == BMname):
							try:
								bpy.ops.object.modifier_apply(apply_as='DATA', modifier=BMname)
							except:
								bpy.ops.object.modifier_remove(modifier=BMname)
								exc_type, exc_value, exc_traceback = sys.exc_info()
								self.report({'ERROR'}, str(exc_value))

				bpy.ops.object.select_all(action='TOGGLE')
			else:
				if self.ObjectMode or self.ProfileMode:
					for mb in self.CurrentObj.modifiers:
						if (mb.type == 'SOLIDIFY') and (mb.name == "CT_SOLIDIFY"):
							try:
								bpy.ops.object.modifier_apply(apply_as='DATA', modifier="CT_SOLIDIFY")
							except:
								exc_type, exc_value, exc_traceback = sys.exc_info()
								self.report({'ERROR'}, str(exc_value))

				# Rebool
				Rebool(context, self)

				# Test if not empty object
				if context.selected_objects[0]:
					rebool_RT = context.selected_objects[0]
					if len(rebool_RT.data.vertices) > 0:
						# Create Bevel for new objects
						if BevelAO:
							print("CREATE BEVEL : ", rebool_RT)
							CreateBevel(context, context.selected_objects[0])
						UndoAdd(self, "REBOOL", context.selected_objects[0])

						context.scene.cursor.location = ActiveObj.location
						bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
					else:
						bpy.ops.object.delete(use_global=False)

				context.scene.cursor.location = CursorLocation

				if self.ObjectMode:
					context.view_layer.objects.active = self.ObjectBrush
				if self.ProfileMode:
					context.view_layer.objects.active = self.ProfileBrush

			if self.DontApply is False:
				# Apply booleans
				BMname = "CT_" + self.CurrentObj.name
				for mb in ActiveObj.modifiers:
					if (mb.type == 'BOOLEAN') and (mb.name == BMname):
						try:
							bpy.ops.object.modifier_apply(apply_as='DATA', modifier=BMname)
						except:
							bpy.ops.object.modifier_remove(modifier=BMname)
							exc_type, exc_value, exc_traceback = sys.exc_info()
							self.report({'ERROR'}, str(exc_value))
				# Get new objects created with rebool operations
				if len(context.selected_objects) > 0:
					if self.shift is True:
						# Get the last object selected
						lastSelected.append(context.selected_objects[0])

		context.scene.cursor.location = CursorLocation

		if self.DontApply is False:
			# Remove cut object
			if (self.ObjectMode is False) and (self.ProfileMode is False):
				if len(context.selected_objects) > 0:
					bpy.ops.object.select_all(action='TOGGLE')
				bpy.data.objects[self.CurrentObj.name].select_set(True)
				bpy.ops.object.delete(use_global=False)
			else:
				if self.ObjectMode:
					self.ObjectBrush.display_type = self.ObjectBrush_DT

		if len(context.selected_objects) > 0:
			bpy.ops.object.select_all(action='TOGGLE')

		# Select cut objects
		for obj in lastSelected:
			bpy.data.objects[obj.name].select_set(True)

		for ActiveObj in ActiveObjList:
			bpy.data.objects[ActiveObj.name].select_set(True)
			context.view_layer.objects.active = ActiveObj
		# Update bevel
		list_act_obj = context.selected_objects.copy()
		if self.Auto_BevelUpdate:
			update_bevel(context)

		# Re-select initial objects
		bpy.ops.object.select_all(action='TOGGLE')
		if self.ObjectMode:
			# Re-select brush
			self.ObjectBrush.select_set(True)
		for ActiveObj in ActiveObjList:
			bpy.data.objects[ActiveObj.name].select_set(True)
			context.view_layer.objects.active = ActiveObj

		# If object has children, set "Wire" draw type
		if self.ObjectBrush is not None:
			if len(self.ObjectBrush.children) > 0:
				self.ObjectBrush.display_type = "WIRE"
		if self.ProfileMode:
			self.ProfileBrush.display_type = "WIRE"

		if bLocalView:
			bpy.ops.view3d.localview()

		# Reset variables
		self.CutMode = False
		self.mouse_path.clear()
		self.mouse_path = [(0, 0), (0, 0)]

		self.ForceRebool = False


class CarverProperties(bpy.types.PropertyGroup):
	DepthCursor: BoolProperty(
		name="DepthCursor",
		default=False
	)
	OInstanciate: BoolProperty(
		name="Obj_Instantiate",
		default=False
	)
	ORandom: BoolProperty(
		name="Random_Rotation",
		default=False
	)
	DontApply: BoolProperty(
		name="Dont_Apply",
		default=False
	)
	nProfile: IntProperty(
		name="Num_Profile",
		default=0
	)


def register():
	from bpy.utils import register_class
	bpy.utils.register_class(CARVER_OT_operator)
	bpy.utils.register_class(CarverProperties)
	bpy.types.Scene.mesh_carver = bpy.props.PointerProperty(type=CarverProperties)

def unregister():
	from bpy.utils import unregister_class
	bpy.utils.unregister_class(CarverProperties)
	bpy.utils.unregister_class(CARVER_OT_operator)
	del bpy.types.Scene.mesh_carver
