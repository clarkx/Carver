import bpy
import bgl
import blf
import bpy_extras
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader
from math import(
	cos,
	sin,
	)

from .carver_utils import (
	draw_circle,
	draw_shader
	)

from mathutils import (
	Color,
	Euler,
	Vector,
	Quaternion,
)

# Draw Text (Center position)
def DrawCenterText(text, xt, yt, Size, colors, self):
	font_id = 0
	# Offset Shadow
	Sshadow_x = 2
	Sshadow_y = -2

	blf.size(font_id, Size, 72)
	blf.position(font_id, xt + Sshadow_x - blf.dimensions(font_id, text)[0] / 2, yt + Sshadow_y, 0)
	blf.color(font_id, 0.0, 0.0, 0.0, 1.0)

	blf.draw(font_id, text)
	blf.position(font_id, xt - blf.dimensions(font_id, text)[0] / 2, yt, 0)
	if colors is not None:
		mcolor = Color((colors[0], colors[1], colors[2]))
		blf.color(font_id,mcolor.r, mcolor.g, mcolor.b, 1.0)
	else:
		blf.color(font_id,1.0, 1.0, 1.0, 1.0)
	blf.draw(font_id, text)


# Draw text (Left position)
def DrawLeftText(text, xt, yt, Size, colors, self):
	font_id = 0
	# Offset Shadow
	Sshadow_x = 2
	Sshadow_y = -2

	blf.size(font_id, Size, 72)
	blf.position(font_id, xt + Sshadow_x, yt + Sshadow_y, 0)
	blf.color(font_id, 0.0, 0.0, 0.0, 1.0)
	blf.draw(font_id, text)
	blf.position(font_id, xt, yt, 0)
	if colors is not None:
		mcolor = Color((colors[0], colors[1], colors[2]))
		blf.color(font_id, mcolor.r, mcolor.g, mcolor.b, 1.0)
	else:
		blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
	blf.draw(font_id, text)


# Draw text (Right position)
def DrawRightText(text, xt, yt, Size, colors, self):
	font_id = 0
	# Offset Shadow
	Sshadow_x = 2
	Sshadow_y = -2

	blf.size(font_id, Size, 72)
	blf.position(font_id, xt + Sshadow_x - blf.dimensions(font_id, text)[0], yt + Sshadow_y, 0)
	blf.color(font_id, 0.0, 0.0, 0.0, 1.0)
	blf.draw(font_id, text)
	blf.position(font_id, xt - blf.dimensions(font_id, text)[0], yt, 0)
	if colors is not None:
		mcolor = Color((colors[0], colors[1], colors[2]))
		blf.color(font_id, mcolor.r, mcolor.g, mcolor.b, 1.0)
	else:
		blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
	blf.draw(font_id, text)


# Opengl draws
def draw_callback_px(self, context):
	font_id = 0
	region = context.region

	# Cut Type
	RECTANGLE = 0
	LINE = 1
	CIRCLE = 2
	self.carver_prefs = context.preferences.addons[__package__].preferences
	# Width screen
	overlap = context.preferences.system.use_region_overlap
	t_panel_width = 0
	if overlap:
		for region in context.area.regions:
			if region.type == 'TOOLS':
				t_panel_width = region.width

	# Initial position
	xt = int(region.width / 2.0)
	yt = 130
	if region.width >= 850:
		xt = int(region.width / 2.0)
		yt = 150

	# Command Display
	if self.CreateMode and ((self.ObjectMode is False) and (self.ProfileMode is False)):
		BooleanMode = "Create"
	else:
		if self.ObjectMode or self.ProfileMode:
			BooleanType = "Difference) [T]" if self.BoolOps == self.difference else "Union) [T]"
			BooleanMode = \
				"Object Brush (" + BooleanType if self.ObjectMode else "Profil Brush (" + BooleanType
		else:
			BooleanMode = \
				"Difference" if (self.shift is False) and (self.ForceRebool is False) else "Rebool"

	UIColor = (0.992, 0.5518, 0.0, 1.0)

	# Display boolean mode
	text_size = 40 if region.width >= 850 else 20
	DrawCenterText(BooleanMode, xt, yt, text_size, UIColor, self)

	# Separator (Line)
	LineWidth = 75
	if region.width >= 850:
		LineWidth = 140

	coords = [(int(xt - LineWidth), yt - 8), (int(xt + LineWidth), yt - 8)]
	draw_shader(self, UIColor, 1, 'LINES', coords, self.carver_prefs.LineWidth)


	# Text position
	xt = xt - blf.dimensions(font_id, "Difference")[0] / 2 + 80

	# Primitives type
	PrimitiveType = "Rectangle "
	if self.CutType == CIRCLE:
		PrimitiveType = "Circle "
	if self.CutType == LINE:
		PrimitiveType = "Line "

	# Variables according to screen size
	IFontSize = 12
	yInterval = 20
	yCmd = yt - 30

	if region.width >= 850:
		IFontSize = 18
		yInterval = 25

	# Color
	Color0 = None
	Color1 = UIColor

	# Help Display
	if (self.ObjectMode is False) and (self.ProfileMode is False):
		TypeStr = "Cut Type [Space] : "
		if self.CreateMode:
			TypeStr = "Type [Space] : "
		blf.size(font_id, IFontSize, 72)
		OpsStr = TypeStr + PrimitiveType
		TotalWidth = blf.dimensions(font_id, OpsStr)[0]
		xLeft = region.width / 2 - TotalWidth / 2
		xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
		DrawLeftText(TypeStr, xLeft, yCmd, IFontSize, Color0, self)
		DrawLeftText(PrimitiveType, xLeftP, yCmd, IFontSize, Color1, self)

		# Depth Cursor
		TypeStr = "Cursor Depth [" + self.carver_prefs.Key_Depth + "] : "
		BoolStr = "(ON)" if self.snapCursor else "(OFF)"
		OpsStr = TypeStr + BoolStr

		TotalWidth = blf.dimensions(font_id, OpsStr)[0]
		xLeft = region.width / 2 - TotalWidth / 2
		xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
		DrawLeftText(TypeStr, xLeft, yCmd - yInterval, IFontSize, Color0, self)
		DrawLeftText(BoolStr, xLeftP, yCmd - yInterval, IFontSize, Color1, self)

		# Close poygonal shape
		if self.CreateMode:
			TypeStr = "Close [" + self.carver_prefs.Key_Close + "] : "
			BoolStr = "(ON)" if self.Closed else "(OFF)"
			OpsStr = TypeStr + BoolStr

			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * 2, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * 2, IFontSize, Color1, self)

		if self.CreateMode is False:
			# Apply Booleans
			TypeStr = "Apply Operations [" + self.carver_prefs.Key_Apply + "] : "
			BoolStr = "(OFF)" if self.DontApply else "(ON)"
			OpsStr = TypeStr + BoolStr

			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * 2, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * 2, IFontSize, Color1, self)

			# Auto update for bevel
			TypeStr = "Bevel Update [" + self.carver_prefs.Key_Update + "] : "
			BoolStr = "(ON)" if self.Auto_BevelUpdate else "(OFF)"
			OpsStr = TypeStr + BoolStr

			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * 3, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * 3, IFontSize, Color1, self)

		# Subdivisions
		if self.CutType == CIRCLE:
			y = yCmd - yInterval * 4 if self.CreateMode is False else yCmd - yInterval * 2
			TypeStr = "Subdivisions [" + self.carver_prefs.Key_Subrem + "][" + self.carver_prefs.Key_Subadd + "] : "
			BoolStr = str((int(360 / self.stepAngle[self.step])))
			OpsStr = TypeStr + BoolStr
			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
			DrawLeftText(TypeStr, xLeft, y, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, y, IFontSize, Color1, self)

	else:
		# INSTANTIATE:
		TypeStr = "Instantiate [" + self.carver_prefs.Key_Instant + "] : "
		BoolStr = "(ON)" if self.Instantiate else "(OFF)"
		OpsStr = TypeStr + BoolStr

		blf.size(font_id, IFontSize, 72)
		TotalWidth = blf.dimensions(font_id, OpsStr)[0]
		xLeft = region.width / 2 - TotalWidth / 2
		xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
		DrawLeftText(TypeStr, xLeft, yCmd, IFontSize, Color0, self)
		DrawLeftText(BoolStr, xLeftP, yCmd, IFontSize, Color1, self)

		# RANDOM ROTATION:
		if self.alt:
			TypeStr = "Random Rotation [" + self.carver_prefs.Key_Randrot + "] : "
			BoolStr = "(ON)" if self.RandomRotation else "(OFF)"
			OpsStr = TypeStr + BoolStr

			blf.size(font_id, IFontSize, 72)
			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval, IFontSize, Color1, self)

		# THICKNESS:
		if self.BrushSolidify:
			TypeStr = "Thickness [" + self.carver_prefs.Key_Depth + "] : "
			if self.ProfileMode:
				BoolStr = str(round(self.ProfileBrush.modifiers["CT_SOLIDIFY"].thickness, 2))
			if self.ObjectMode:
				BoolStr = str(round(self.ObjectBrush.modifiers["CT_SOLIDIFY"].thickness, 2))
			OpsStr = TypeStr + BoolStr
			blf.size(font_id, IFontSize, 72)
			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]

			self_alt_y = 2 if self.alt else 1
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * self_alt_y, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * self_alt_y, IFontSize, Color1, self)

		# BRUSH DEPTH:
		if (self.ObjectMode):
			TypeStr = "Carve Depth [" + self.carver_prefs.Key_Depth + "] : "
			BoolStr = str(round(self.ObjectBrush.data.vertices[0].co.z, 2))
			OpsStr = TypeStr + BoolStr

			blf.size(font_id, IFontSize, 72)
			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]

			self_alt_y = 2 if self.alt else 1
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * self_alt_y, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * self_alt_y, IFontSize, Color1, self)

			TypeStr = "Brush Depth [" + self.carver_prefs.Key_BrushDepth + "] : "
			BoolStr = str(round(self.BrushDepthOffset, 2))
			OpsStr = TypeStr + BoolStr

			blf.size(font_id, IFontSize, 72)
			TotalWidth = blf.dimensions(font_id, OpsStr)[0]
			xLeft = region.width / 2 - TotalWidth / 2
			xLeftP = xLeft + blf.dimensions(font_id, TypeStr)[0]

			self_alt_y = 3 if self.alt else 2
			DrawLeftText(TypeStr, xLeft, yCmd - yInterval * self_alt_y, IFontSize, Color0, self)
			DrawLeftText(BoolStr, xLeftP, yCmd - yInterval * self_alt_y, IFontSize, Color1, self)

	if region.width >= 850:
		if self.AskHelp is False:
			xrect = 40
			yrect = 40
			coords = [(xrect, yrect), (xrect+90, yrect), (xrect+90, yrect+25), (xrect, yrect+25)]

			draw_shader(self, (0.0, 0.0, 0.0),  0.3, 'TRI_FAN', coords, self.carver_prefs.LineWidth)
			DrawLeftText("[" + self.carver_prefs.Key_Help + "] for help", xrect + 10, yrect + 8, 13, None, self)
		else:
			xHelp = 30 + t_panel_width
			yHelp = 80
			Help_FontSize = 12
			Help_Interval = 14
			if region.width >= 850:
				Help_FontSize = 15
				Help_Interval = 20
				yHelp = 220

			if self.ObjectMode or self.ProfileMode:
				if self.ProfileMode:
					DrawLeftText("[" + self.carver_prefs.Key_Brush + "]", xHelp, yHelp +
								Help_Interval * 2, Help_FontSize, UIColor, self)
					DrawLeftText(": Object Mode", 150 + t_panel_width, yHelp +
								Help_Interval * 2, Help_FontSize, None, self)
				else:
					DrawLeftText("[" + self.carver_prefs.Key_Brush + "]", xHelp, yHelp +
								Help_Interval * 2, Help_FontSize, UIColor, self)
					DrawLeftText(": Return", 150 + t_panel_width, yHelp +
								Help_Interval * 2, Help_FontSize, None, self)
			else:
				DrawLeftText("[" + self.carver_prefs.Key_Brush + "]", xHelp, yHelp +
							Help_Interval * 2, Help_FontSize, UIColor, self)
				DrawLeftText(": Profil Brush", 150 + t_panel_width, yHelp +
							Help_Interval * 2, Help_FontSize, None, self)
				DrawLeftText("[Ctrl + LMB]", xHelp, yHelp - Help_Interval * 6,
							Help_FontSize, UIColor, self)
				DrawLeftText(": Move Cursor", 150 + t_panel_width, yHelp -
							Help_Interval * 6, Help_FontSize, None, self)

			if (self.ObjectMode is False) and (self.ProfileMode is False):
				if self.CreateMode is False:
					DrawLeftText("[" + self.carver_prefs.Key_Create + "]", xHelp,
								 yHelp + Help_Interval, Help_FontSize, UIColor, self)
					DrawLeftText(": Create geometry", 150 + t_panel_width,
								 yHelp + Help_Interval, Help_FontSize, None, self)
				else:
					DrawLeftText("[" + self.carver_prefs.Key_Create + "]", xHelp,
								 yHelp + Help_Interval, Help_FontSize, UIColor, self)
					DrawLeftText(": Cut", 150 + t_panel_width, yHelp + Help_Interval,
								Help_FontSize, None, self)

				if self.CutType == RECTANGLE:
					DrawLeftText("MouseMove", xHelp, yHelp, Help_FontSize, UIColor, self)
					DrawLeftText("[Alt]", xHelp, yHelp - Help_Interval, Help_FontSize, UIColor, self)
					DrawLeftText(": Dimension", 150 + t_panel_width, yHelp, Help_FontSize, None, self)
					DrawLeftText(": Move all", 150 + t_panel_width, yHelp - Help_Interval,
								Help_FontSize, None, self)

				if self.CutType == CIRCLE:
					DrawLeftText("MouseMove", xHelp, yHelp, Help_FontSize, UIColor, self)
					DrawLeftText("[Alt]", xHelp, yHelp - Help_Interval, Help_FontSize, UIColor, self)
					DrawLeftText("[" + self.carver_prefs.Key_Subrem + "] [" + context.scene.Key_Subadd + "]",
								xHelp, yHelp - Help_Interval * 2, Help_FontSize, UIColor, self)
					DrawLeftText("[Ctrl]", xHelp, yHelp - Help_Interval * 3, Help_FontSize, UIColor, self)
					DrawLeftText(": Rotation and Radius", 150 + t_panel_width, yHelp, Help_FontSize, None, self)
					DrawLeftText(": Move all", 150 + t_panel_width, yHelp - Help_Interval,
								Help_FontSize, None, self)
					DrawLeftText(": Subdivision", 150 + t_panel_width, yHelp -
								Help_Interval * 2, Help_FontSize, None, self)
					DrawLeftText(": Incremental rotation", 150 + t_panel_width,
								yHelp - Help_Interval * 3, Help_FontSize, None, self)

				if self.CutType == LINE:
					DrawLeftText("MouseMove", xHelp, yHelp, Help_FontSize, UIColor, self)
					DrawLeftText("[Alt]", xHelp, yHelp - Help_Interval, Help_FontSize, UIColor, self)
					DrawLeftText("[Space]", xHelp, yHelp - Help_Interval * 2, Help_FontSize, UIColor, self)
					DrawLeftText("[Ctrl]", xHelp, yHelp - Help_Interval * 3, Help_FontSize, UIColor, self)
					DrawLeftText(": Dimension", 150 + t_panel_width, yHelp, Help_FontSize, None, self)
					DrawLeftText(": Move all", 150 + t_panel_width, yHelp - Help_Interval,
								Help_FontSize, None, self)
					DrawLeftText(": Validate", 150 + t_panel_width, yHelp -
								 Help_Interval * 2, Help_FontSize, None, self)
					DrawLeftText(": Incremental", 150 + t_panel_width, yHelp -
								Help_Interval * 3, Help_FontSize, None, self)
					if self.CreateMode:
						DrawLeftText("[" + self.carver_prefs.Key_Subadd + "]", xHelp, yHelp -
									Help_Interval * 4, Help_FontSize, UIColor, self)
						DrawLeftText(": Close geometry", 150 + t_panel_width, yHelp -
									Help_Interval * 4, Help_FontSize, None, self)
			else:
				DrawLeftText("[Space]", xHelp, yHelp + Help_Interval, Help_FontSize, UIColor, self)
				DrawLeftText(": Difference", 150 + t_panel_width, yHelp + Help_Interval,
							Help_FontSize, None, self)
				DrawLeftText("[Shift][Space]", xHelp, yHelp, Help_FontSize, UIColor, self)
				DrawLeftText(": Rebool", 150 + t_panel_width, yHelp, Help_FontSize, None, self)
				DrawLeftText("[Alt][Space]", xHelp, yHelp - Help_Interval, Help_FontSize, UIColor, self)
				DrawLeftText(": Duplicate", 150 + t_panel_width, yHelp - Help_Interval,
							Help_FontSize, None, self)
				DrawLeftText("[" + self.carver_prefs.Key_Scale + "]", xHelp, yHelp -
							Help_Interval * 2, Help_FontSize, UIColor, self)
				DrawLeftText(": Scale", 150 + t_panel_width, yHelp - Help_Interval * 2,
							Help_FontSize, None, self)
				DrawLeftText("[LMB][Move]", xHelp, yHelp - Help_Interval * 3, Help_FontSize, UIColor, self)
				DrawLeftText(": Rotation", 150 + t_panel_width, yHelp - Help_Interval * 3,
							Help_FontSize, None, self)
				DrawLeftText("[Ctrl][LMB][Move]", xHelp, yHelp - Help_Interval * 4,
							Help_FontSize, UIColor, self)
				DrawLeftText(": Step Angle", 150 + t_panel_width, yHelp - Help_Interval * 4,
							Help_FontSize, None, self)
				if self.ProfileMode:
					DrawLeftText("[" + self.carver_prefs.Key_Subadd + "][" + self.carver_prefs.Key_Subrem + "]",
								xHelp, yHelp - Help_Interval * 5, Help_FontSize, UIColor, self)
					DrawLeftText(": Previous or Next Profile", 150 + t_panel_width,
								 yHelp - Help_Interval * 5, Help_FontSize, None, self)
				DrawLeftText("[ARROWS]", xHelp, yHelp - Help_Interval * 6, Help_FontSize, UIColor, self)
				DrawLeftText(": Create / Delete rows or columns", 150 + t_panel_width,
							yHelp - Help_Interval * 6, Help_FontSize, None, self)
				DrawLeftText("[" + self.carver_prefs.Key_Gapy + "][" + self.carver_prefs.Key_Gapx + "]",
							 xHelp, yHelp - Help_Interval * 7, Help_FontSize, UIColor, self)
				DrawLeftText(": Gap between rows or columns", 150 + t_panel_width,
							yHelp - Help_Interval * 7, Help_FontSize, None, self)


	if self.ProfileMode:
		xrect = region.width - t_panel_width - 80
		yrect = 80
		coords = [(xrect, yrect), (xrect+60, yrect), (xrect+60, yrect-60), (xrect, yrect-60)]

		#Draw rectangle background in the lower right
		draw_shader(self, (0.0, 0.0, 0.0),  0.3, 'TRI_FAN', coords, self.carver_prefs.LineWidth)

		WidthProfil = 50
		location = Vector((region.width - t_panel_width - WidthProfil, 50, 0))
		ProfilScale = 20.0
		coords = []
		mesh = bpy.data.meshes[self.Profils[self.nProfil][0]]
		mesh.calc_loop_triangles()
		vertices = np.empty((len(mesh.vertices), 3), 'f')
		indices = np.empty((len(mesh.loop_triangles), 3), 'i')
		mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
		mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))

		for idx, vals in enumerate(vertices):
			coords.append([
			vals[0] * ProfilScale + location.x,
			vals[1] * ProfilScale + location.y,
			vals[2] * ProfilScale + location.z
			])

		#Draw the silhouette of the mesh
		draw_shader(self, UIColor,  0.5, 'TRIS', coords, self.carver_prefs.LineWidth, indices=indices)


	if self.CutMode:

		if len(self.mouse_path) > 1:
			x0 = self.mouse_path[0][0]
			y0 = self.mouse_path[0][1]
			x1 = self.mouse_path[1][0]
			y1 = self.mouse_path[1][1]

		# Cut rectangle
		if self.CutType == RECTANGLE:
			coords = [
			(x0 + self.xpos, y0 + self.ypos), (x1 + self.xpos, y0 + self.ypos), \
			(x1 + self.xpos, y1 + self.ypos), (x0 + self.xpos, y1 + self.ypos), \
			(x0 + self.xpos, y0 + self.ypos)]
			indices = ((0, 1, 2), (2, 0, 3))

			draw_shader(self, UIColor, 1, 'LINE_LOOP', coords, self.carver_prefs.LineWidth)

			#Draw points
			draw_shader(self, UIColor, 1, 'POINTS', coords, self.carver_prefs.LineWidth)

			if self.shift or self.CreateMode:
				draw_shader(self, UIColor, 0.5, 'TRIS', coords, self.carver_prefs.LineWidth, indices=indices)

		# Cut Line
		elif self.CutType == LINE:

			coords = []
			indices = []

			for idx, vals in enumerate(self.mouse_path):
				coords.append([vals[0] + self.xpos, vals[1] + self.ypos])
				indices.append([idx])

			#Draw lines
			draw_shader(self, UIColor, 1.0, 'LINE_LOOP', coords, self.carver_prefs.LineWidth)

			#Draw points
			draw_shader(self, UIColor, 1.0, 'POINTS', coords, self.carver_prefs.LineWidth)

			#Draw polygon
			if (self.shift) or (self.CreateMode and self.Closed):
				draw_shader(self, UIColor, 0.5, 'TRI_FAN', coords, self.carver_prefs.LineWidth)

		# Circle Cut
		elif self.CutType == CIRCLE:
			radius = self.mouse_path[1][0] - self.mouse_path[0][0]
			steps = int(360 / self.stepAngle[self.step])
			DEG2RAD = 3.14159 / (180.0 / self.stepAngle[self.step])

			if self.ctrl:
				self.step_rotation = (self.mouse_path[1][1] - self.mouse_path[0][1]) / 25
				rotate_circle = (3.14159 / (360.0 / 60.0)) * int(self.step_rotation)
			else:
				rotate_circle = (self.mouse_path[1][1] - self.mouse_path[0][1]) / 50

			circle_coords, line_coords, indices = draw_circle(self, x0, y0)
			draw_shader(self, UIColor, 1.0, 'LINE_LOOP', line_coords, self.carver_prefs.LineWidth)

			if self.shift or self.CreateMode:
				draw_shader(self, UIColor, 0.5, 'TRIS', circle_coords, self.carver_prefs.LineWidth, indices=indices)

	if self.ObjectMode or self.ProfileMode:
		if self.ShowCursor:
			region = context.region
			rv3d = context.space_data.region_3d

			if self.ObjectMode:
				ob = self.ObjectBrush
			if self.ProfileMode:
				ob = self.ProfileBrush
			mat = ob.matrix_world

			# 50% alpha, 2 pixel width line
			bgl.glEnable(bgl.GL_BLEND)

			bbox = [mat @ Vector(b) for b in ob.bound_box]

			if self.shift:
				gl_line_width = 4
				UIColor = (0.5, 1.0, 0.0, 1.0)
			else:
				gl_line_width = 2
				UIColor = (1.0, 0.8, 0.0, 1.0)

			line_coords = []
			idx = 0
			CRadius = ((bbox[7] - bbox[0]).length) / 2
			for i in range(int(len(self.CLR_C) / 3)):
				vector3d = (self.CLR_C[idx * 3] * CRadius + self.CurLoc.x, self.CLR_C[idx * 3 + 1] *
							CRadius + self.CurLoc.y, self.CLR_C[idx * 3 + 2] * CRadius + self.CurLoc.z)
				vector2d = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, vector3d)
				if vector2d is not None:
					line_coords.append((vector2d[0], vector2d[1]))
				idx += 1

			draw_shader(self, UIColor, 1.0, 'LINE_LOOP', line_coords, gl_line_width)

			# Object display
			if self.quat_rot is not None:
				ob.location = self.CurLoc
				v = Vector()
				v.x = v.y = 0.0
				v.z = self.BrushDepthOffset
				ob.location += self.quat_rot @ v

				e = Euler()
				e.x = 0.0
				e.y = 0.0
				e.z = self.aRotZ / 25.0

				qe = e.to_quaternion()
				qRot = self.quat_rot @ qe
				ob.rotation_mode = 'QUATERNION'
				ob.rotation_quaternion = qRot
				ob.rotation_mode = 'XYZ'

				if self.ProfileMode:
					if self.ProfileBrush is not None:
						self.ProfileBrush.location = self.CurLoc
						self.ProfileBrush.rotation_mode = 'QUATERNION'
						self.ProfileBrush.rotation_quaternion = qRot
						self.ProfileBrush.rotation_mode = 'XYZ'

	# Opengl defaults
	bgl.glLineWidth(1)
	bgl.glDisable(bgl.GL_BLEND)
	# bgl.color(font_id, 0.0, 0.0, 0.0, 1.0)
	# bgl.glDisable(bgl.GL_POINT_SMOOTH)
