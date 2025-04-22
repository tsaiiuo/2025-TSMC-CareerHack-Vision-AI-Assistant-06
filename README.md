# AI Vision Task Assistant

## Overview
AI Vision Task Assistant is an innovative project aimed at simplifying and automating visual inspection tasks. Utilizing cutting-edge AI tools and models, it addresses common challenges faced by vision engineers, such as tedious coding, frequent adjustments, and error-prone processes.

## Team Members
- 蔡哲偉
- 蔡旻桓
- 楊淨雯
- 賴柏叡

## User Story

### Problem
Vision engineers typically face significant challenges:
- Writing code line-by-line to detect chip defects.
- Repeatedly adjusting algorithms with changing product specifications.
- High risk of errors and inefficiency in manual methods.

### Solution
Our Vision Supported Agent significantly simplifies these tasks through an intuitive AI-driven interface:

- **Natural Language Object Detection:** Utilizes Dino for object detection based on descriptive prompts.
- **Rotation Correction:** Employs OpenCV (cv2) for image rotation adjustments.
- **Universal Segmentation:** Incorporates SAM2 (Segment Anything Model) for precise image segmentation.
- **Image Generation & Inpainting:** Leverages Diffusion models to generate and modify images seamlessly.
- **Image Creation:** Integrates VertexAI for high-quality image synthesis.

## System Architecture
The system integrates several advanced technologies:
- **Dino:** Object detection using natural language prompts.
- **OpenCV (cv2):** Image manipulation and rotation handling.
- **SAM2:** General-purpose segmentation.
- **Diffusion Models:** Image generation and inpainting.
- **VertexAI:** Advanced image generation capabilities.

## Prompt Planning
Our approach to prompt planning enhances AI interaction efficiency and accuracy:

- **Few-Shot Prompting:** Leveraging limited examples effectively.
- **Chain-of-Note Prompting:** Encourages systematic reasoning and reflection by the model.
- **Knowledge Component:** Automatically identifies relevant examples to reference, creating effective and contextual responses.
- **Reflection Tagging:** Enables the LLM (Large Language Model) to evaluate and reflect on its actions for better accuracy.

## User Interface
- Developed with **Next.js** for a responsive and intuitive UI.
- Features interactive elements such as radar charts to visually represent data and analysis.

## Demo
Watch our demonstration video to understand our AI Vision Task Assistant in action:

[AI Vision Task Assistant Demo Video](http://www.youtube.com/watch?v=XsJ37pzegiA)

## Conclusion
Our AI Vision Task Assistant streamlines the inspection process, significantly reducing manual effort and increasing productivity for vision engineers.

---

Thanks for checking out our project!

