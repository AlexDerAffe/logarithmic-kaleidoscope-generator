# Logarithmic Kaleidoscope

A hardware-accelerated kaleidoscope effect implemented in Python using Pygame and ModernGL, 
inspired by the mathematical concepts from 3Blue1Brown's video on image logarithms.

The project processes an image using a CPU-bound logarithmic transformation (`scipy`), turning scaling into translation, 
and then uses a GPU fragment shader (`GLSL`) to render a smooth, real-time animated kaleidoscope effect.

ACKNOWLEDGEMENTS
Inspiration and mathematical background from the 3Blue1Brown video: How (and why) to take a logarithm of an image.
