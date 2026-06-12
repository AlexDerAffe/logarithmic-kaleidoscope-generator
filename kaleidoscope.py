import sys
import moderngl
import numpy as np
import pygame
from scipy.ndimage import map_coordinates

# Configuration
SCREEN_SIZE = 800
NUM_SEGMENTS = 12.0  # float for the shader
PHI_SPEED = 0.2    # rotation speed
NUM_IMAGES_X = 5  # number of repeated images in x direction for the log transform, try the log transform for itself to see what it does.
NUM_IMAGES_Y = 5  # similar to NUM_IMAGES_X.
ORIGINAL_IMAGE_PATH = "cover4.jpeg"  # source picture


# a cool transformation, inspired by a 3Blue1Brown video, that applies a Logarithmic transformation. 
# The video is link is https://youtu.be/ldxFjLJ3rVY?si=G46XBH7olgWBBdDL for further explanation and visualizations of the transformation.
def apply_log_transform(source_surface, target_width=800, target_height=800, total_images_x=5, total_images_y=5):
    # covert to numpy array for processing
    source_array = pygame.surfarray.array3d(source_surface)
    src_h, src_w, channels = source_array.shape
    cx, cy = src_w / 2, src_h / 2
    
    # create grid of target coordinates
    x_indices, y_indices = np.meshgrid(
        np.arange(target_width), 
        np.arange(target_height), 
        indexing='ij'
    )
    
    # max radius for the kaleidoscope effect, based on the center of the source image
    max_r = min(cx, cy)
    
    # clip_threshold clips the inner part of the image to avoid extreme distortion and maintain more detail in the outer areas.
    clip_threshold = 0.1
    
    # periodic mirroring for x-axis (radial direction)
    x_linear = x_indices / (target_width / total_images_x)
    x_mirrored = np.abs((x_linear % 2) - 1)
    
    # starting from the log of the radius allows us to have more detail in the outer areas of the kaleidoscope,
    # while still maintaining a smooth transition towards the center
    # The clip_threshold determines how much of the inner part of the image is "clipped" or compressed,
    # which can help to reduce distortion and maintain more recognizable features in the outer areas.
    log_min = np.log(clip_threshold * max_r + 1)
    log_max = np.log(max_r + 1)
    
    # only the part of the radius that is above the clip_threshold is transformed logarithmically,
    # while the inner part is compressed to fit within the clip_threshold
    # This way, we can maintain more detail in the outer areas of the kaleidoscope,
    # while still having a smooth transition towards the center.
    log_r = log_min + (1 - x_mirrored) * (log_max - log_min)
    r = np.exp(log_r)
    
    # periodic mirroring for y-axis (angular direction)
    y_linear = y_indices / (target_height / total_images_y)
    y_mirrored = np.abs((y_linear % 2) - 1)
    theta = y_mirrored * 2 * np.pi
    
    # transforming polar coordinates back to Cartesian coordinates for sampling the source image
    orig_x = cx + r * np.cos(theta)
    orig_y = cy + r * np.sin(theta)
    
    # safety clamp to ensure we don't sample outside the source image boundaries
    orig_x = np.clip(orig_x, 0, src_w - 1)
    orig_y = np.clip(orig_y, 0, src_h - 1)
    
    # billinear interpolation to sample the source image at the calculated coordinates
    dest_array = np.zeros((target_width, target_height, channels), dtype=np.uint8)
    coordinates = np.array([orig_x.ravel(), orig_y.ravel()])
    
    for c in range(channels):
        channel_interpolated = map_coordinates(
            source_array[:, :, c], 
            coordinates, 
            order=1, 
            mode='reflect',
            prefilter=False
        )
        dest_array[:, :, c] = channel_interpolated.reshape(target_width, target_height)
    
    return pygame.surfarray.make_surface(dest_array)


def main():
    pygame.init()
    
    # we're using OPENGL and DOUBLEBUF to enable hardware acceleration and smooth rendering
    pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Kaleidoskop – Hardwarebeschleunigt via GLSL")
    clock = pygame.time.Clock()

    # initialize OpenGL context using moderngl
    ctx = moderngl.create_context()

    original_image = pygame.image.load(ORIGINAL_IMAGE_PATH).convert()
    try:
        # the source picture is processed in the log transform, which kind of warps it and makes everything more abstract and trippy
        # try the transformation for yourself to see how it works or watch the 3B1B video linked above.
        source_img = apply_log_transform(original_image, SCREEN_SIZE, SCREEN_SIZE, total_images_x=NUM_IMAGES_X, total_images_y=NUM_IMAGES_Y)
    except (FileNotFoundError, pygame.error):
        # if no image is found, create a simple placeholder surface with a circle
        source_img = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        source_img.fill((30, 50, 150))
        pygame.draw.circle(source_img, (220, 180, 50), (SCREEN_SIZE // 2, SCREEN_SIZE // 2), 250, 40)

    source_img = pygame.transform.smoothscale(source_img, (SCREEN_SIZE, SCREEN_SIZE))
    texture_data = pygame.image.tostring(source_img, "RGB", True)
    
    # create an OpenGL texture from the processed image data
    texture = ctx.texture((SCREEN_SIZE, SCREEN_SIZE), 3, texture_data)
    texture.repeat_x = True
    texture.repeat_y = True
    texture.use(location=0)

    # GLSL SHADER CODE
    # Vertex Shader
    vertex_shader = """
    #version 330
    in vec2 in_vert;
    out vec2 uvs;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        uvs = in_vert * 0.5 + 0.5; // Von (-1,1) zu (0,1) Texturkoordinaten
    }
    """

    # Fragment Shader
    fragment_shader = """
    #version 330
    in vec2 uvs;
    out vec4 fragColor;

    uniform sampler2D u_texture;
    uniform float u_num_segments;
    uniform float u_phi;
    uniform vec2 u_source_center;

    const float PI = 3.14159265359;

    void main() {
        // centered coordinates (0,0) in the middle of the screen, range from -0.5 to 0.5
        vec2 p = uvs - vec2(0.5);
        float radius = length(p);

        // everything outside the radius of 0.5 (the corners) is black, to create a circular kaleidoscope effect
        if (radius > 0.5) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        // calculate polar coordinates (angle and radius) for the current pixel
        float angle = atan(p.y, p.x);
        if (angle < 0.0) angle += 2.0 * PI;

        // create segments and mirror every second segment to create the kaleidoscope effect
        float segment_angle = (2.0 * PI) / u_num_segments;
        float segment_idx = floor(angle / segment_angle);
        float local_angle = angle - (segment_idx * segment_angle);
        if (mod(segment_idx, 2.0) == 1.0) {
            local_angle = segment_angle - local_angle;
        }

        // rotates the local angle by phi to create the animation effect, and adds the offset to the source coordinates
        float phi_rad = radians(u_phi);

        // local coordinates in the source image, based on the radius and the rotated local angle
        float src_x_temp = radius * cos(local_angle);
        float src_y_temp = radius * sin(local_angle);

        float src_x = src_x_temp * cos(phi_rad) - src_y_temp * sin(phi_rad) + u_source_center.x;
        float src_y = src_x_temp * sin(phi_rad) + src_y_temp * cos(phi_rad) + u_source_center.y;

        // get pixels from the source texture using the calculated coordinates
        vec2 final_uv = vec2(src_x, src_y);
        fragColor = texture(u_texture, final_uv);
    }
    """

    # compile the shaders and create a shader program
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # create a simple quad that covers the entire screen
    vertices = np.array([-1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0,  1.0], dtype='f4')
    vbo = ctx.buffer(vertices)
    vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])

    # connect the texture and uniform variables to the shader program
    prog['u_texture'].value = 0
    prog['u_num_segments'].value = NUM_SEGMENTS
    
    # rotational axis offset, to create a more dynamic and interesting composition 
    # by shifting the center of the kaleidoscope effect away from the exact center of the screen.
    offset_x = 0.5 + 0.15   # +0.15 is the additional offset to the right, 0.5 would mean its the exact center of the screen
    offset_y = 0.5 - 0.15   # vice versa
    prog['u_source_center'].value = (offset_x, offset_y)

    phi = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # update the rotation angle (phi) for the animation, and pass it to the shader program
        phi = (phi + PHI_SPEED) % 360.0
        prog['u_phi'].value = phi

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()