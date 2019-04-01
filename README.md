# Carver MT (Pixivore) for 2.8 - Beta
Multiple tools to carve or to create objects

Done :
- Refactoring the code:
  - Rewrite the help menu :
    - Add new function : 
      - get_text_info : Return the dimensions of each part of the text
      - draw_string : Draw each string in the text
      
  - Rewrite the bgl :
    - Add a new function : draw_shader
    
  - Delete the old isect_line_plane_v3 function by Ideasman42 :
    - Replace it with the internal function : intersect_line_plane
  
  - Delete the old increment mode:
    - Replace it with the scale and subdivisions of the internal overlay grid
    
- Bug fix
- Add stuff:
  - Add a bgl grid with line cut :
    - When in cut line mode, use the ctrl key to display a little grid align the overlay
    - Use wheel up/down of the mouse to controle the scale of the snap
    - work only in ortho mode
    
  - Add a check_region function:
    - In cut mode, it was possible to draw outside the 3dView. Now, if the cursor is outside the 3dView, the draw mode is blocked and the opengl will be red.
  
