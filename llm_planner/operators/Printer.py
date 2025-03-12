from llm_planner.actor.operator import Operator
from llm_planner.message import Message

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image

from cairosvg import svg2png

import os


class Printer(Operator):

    def __init__(self, output_dir="pdf_files"):
        super().__init__()
        # Directory where PDFs will be stored
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_svg_to_png(self, svg_path, png_path):
        svg2png(url=svg_path, write_to=png_path)

    def convert_to_pdf(self, path, filename):
        # Determine the file extension
        file_extension = os.path.splitext(path)[1].lower()

        file_path = os.path.join(self.output_dir, filename)
        c = canvas.Canvas(file_path, pagesize=letter)

        # Check if the path is an image, text file, or .docx file
        if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            # Handle image files
            fpath = path
            if file_extension == '.svg':
                fpath = path[:-3] + 'png'
                self.convert_to_pdf(path, fpath)
            self.add_image_to_pdf(c, fpath)
        elif file_extension == '.txt':
            # Handle text files
            self.add_text_to_pdf(c, path)
        elif file_extension == '.docx':
            # Handle .docx files
            self.add_docx_to_pdf(c, path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        c.save()
        return file_path

    def add_image_to_pdf(self, c, image_path):
        """Add an image to the PDF."""
        # Get the page size (width and height)
        page_width, page_height = letter

        # Define the image position and size (adjust based on page size)
        image_width = 400
        image_height = 300

        # Ensure the image coordinates are within the page limits
        x = 100  # X coordinate from the left
        y = page_height - image_height - 50  # Y coordinate from the bottom (adjusted for image height)

        # Check if the image fits within the page dimensions
        if y < 0 or x + image_width > page_width:
            raise ValueError(
                "Image position or size is out of range for the page dimensions."
            )

        # Draw the image
        c.drawImage(image_path, x, y, width=image_width, height=image_height)

        # Add a caption below the image
        c.drawString(x, y - 20, "Image inserted.")

    def draw_multiline_text(self,
                            c,
                            text_content,
                            x,
                            y,
                            max_lines=40,
                            line_height=15):
        lines = text_content.splitlines()
        for i, line in enumerate(
                lines[:max_lines]):  # Limit the number of lines
            c.drawString(x, y - (i * line_height), line)
        if len(lines) > max_lines:
            c.drawString(x, y - (max_lines * line_height),
                         "(Truncated for length)")

    def add_text_to_pdf(self, c, text_path):
        """Add text from a text file to the PDF."""
        with open(text_path, 'r') as text_file:
            text_content = text_file.read()
            self.draw_multiline_text(c, text_content, x=100, y=750)

    async def process(self, sender_id, message: Message):
        if message["content"] is not None:
            # If the message contains a filename, use it. Otherwise, use a default name
            filename = message['filename']
            filename = 'output.pdf' if filename is None else filename
            path = message['content']  # Path to the file to be printed

            try:
                # Convert the file (text/image/docx) into a PDF
                pdf_path = self.convert_to_pdf(path, filename)

                # Prepare the response message with the PDF file path
                response_msg = message.spawn()
                response_msg['request_message'] = message
                response_msg['pdf_file'] = pdf_path

                # Send back the response
                self.send(sender_id, response_msg)
            except ValueError as e:
                # Handle unsupported file types
                print(f"Error: {str(e)}")
