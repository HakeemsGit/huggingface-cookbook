import fitz
import os
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

class PDFConverter:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file and return it as a string.
        """
        LOGGER.info(f"Starting text extraction from: {pdf_path}")
        try:
            # Open the PDF file
            with fitz.open(pdf_path) as doc:
                LOGGER.debug(f"Processing PDF with {doc.page_count} pages")
                text_content = []
                
                # Use tqdm for progress bar
                for page_num in tqdm(range(doc.page_count), desc="Extracting pages"):
                    try:
                        page = doc[page_num]
                        text_content.append(page.get_text())
                        LOGGER.debug(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        LOGGER.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                
                LOGGER.info("Successfully completed PDF text extraction")
                return "\n\n".join(text_content)
        except Exception as e:
            LOGGER.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return f"Error extracting text: {str(e)}"

    @staticmethod
    def convert_pdf_directory(pdf_dir="pdfs", output_dir="books"):
        """
        Convert all PDFs in a directory to text files.
        """
        # Create directories if they don't exist
        pdf_path = Path(pdf_dir)
        output_path = Path(output_dir)
        
        pdf_path.mkdir(exist_ok=True)
        output_path.mkdir(exist_ok=True)
        
        # Get list of PDF files
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            LOGGER.warning(f"No PDF files found in {pdf_dir}")
            print(f"Please add PDF files to the '{pdf_dir}' directory and run again.")
            return
        
        print(f"\nFound {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for pdf_file in pdf_files:
            # Create output filename
            output_file = output_path / f"{pdf_file.stem}.txt"
            
            # Skip if already processed
            if output_file.exists():
                LOGGER.info(f"Skipping {pdf_file.name} - already processed")
                continue
            
            print(f"\nProcessing: {pdf_file.name}")
            
            # Extract text
            text_content = PDFConverter.extract_text_from_pdf(str(pdf_file))
            
            # Save to text file
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                LOGGER.info(f"Successfully saved {output_file}")
            except Exception as e:
                LOGGER.error(f"Error saving {output_file}: {str(e)}")

def main():
    print("PDF to Text Converter")
    print("====================")
    
    # Set up directories
    pdf_dir = "pdfs"
    output_dir = "books"
    
    print(f"\nPDF Directory: {pdf_dir}")
    print(f"Output Directory: {output_dir}")
    print("\nStarting conversion...")
    
    # Convert PDFs
    PDFConverter.convert_pdf_directory(pdf_dir, output_dir)
    
    print("\nConversion complete!")
    print(f"Check the '{output_dir}' directory for the converted text files.")

if __name__ == "__main__":
    main()