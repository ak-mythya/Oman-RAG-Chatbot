#!/usr/bin/env python3
"""
Document Ingestion Pipeline – Markdown Only

This class implements a pipeline that:
  1. Converts PDF files to Markdown using Docling.
  2. (Optionally) Cleans the Markdown content.
  3. (Optionally) Inserts generated image captions into the Markdown.
  4. Splits the processed Markdown into overlapping chunks.
  5. Enriches each chunk with metadata and converts them into LangChain Document objects.

Required packages:
    pip install docling nltk pyspellchecker num2words langchain langchain-community markdown
"""

import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Dict

# --- Docling Imports ---
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# --- Text Cleaning and Markdown Imports ---
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from num2words import num2words
import markdown

# --- LangChain Imports ---
from langchain.docstore.document import Document

# --- Ensure Required NLTK Data ---
for pkg in ['stopwords', 'wordnet', 'punkt']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    A pipeline to ingest PDF documents and convert them into LangChain Document objects.
    """

    def __init__(self, input_paths: List[Path], output_dir: Path, clean: bool = True, caption: bool = False):
        """
        Initialize the pipeline.
        :param input_paths: List of PDF file paths.
        :param output_dir: Directory where converted Markdown and images will be stored.
        :param clean: Whether to clean the Markdown content.
        :param caption: Whether to insert image captions into the Markdown.
        """
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.clean = clean
        self.caption = caption
        self.conv_results: List[ConversionResult] = []
        self.markdown_file: Path = None

    # === Part 1: PDF to Markdown Conversion ===

    def export_markdown_only(self, conv_results: Iterable[ConversionResult]) -> None:
        """
        Export Markdown output from conversion results.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                base_name = conv_res.input.file.stem
                md_file = self.output_dir / f"{base_name}.md"
                md_file.write_text(conv_res.document.export_to_markdown(), encoding="utf-8")
                _log.info(f"Exported Markdown: {md_file}")
            else:
                _log.info(f"Conversion failed or partial for: {conv_res.input.file}")

    def export_images(self, conv_results: Iterable[ConversionResult]) -> None:
        """
        Export images from conversion results into an 'images' subfolder.
        """
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        from docling_core.types.doc import PictureItem, TableItem

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                base_name = conv_res.input.file.stem
                table_counter = 0
                picture_counter = 0
                for element, _level in conv_res.document.iterate_items():
                    if isinstance(element, TableItem):
                        table_counter += 1
                        try:
                            image = element.get_image(conv_res.document)
                            if image is not None:
                                image_filename = images_dir / f"{base_name}-table-{table_counter}.png"
                                with open(image_filename, "wb") as fp:
                                    image.save(fp, "PNG")
                                _log.info(f"Exported table image: {image_filename}")
                        except Exception as e:
                            _log.error(f"Error exporting table image for {base_name}-table-{table_counter}: {e}")
                    elif isinstance(element, PictureItem):
                        picture_counter += 1
                        try:
                            image = element.get_image(conv_res.document)
                            if image is not None:
                                image_filename = images_dir / f"{base_name}-picture-{picture_counter}.png"
                                with open(image_filename, "wb") as fp:
                                    image.save(fp, "PNG")
                                _log.info(f"Exported picture image: {image_filename}")
                        except Exception as e:
                            _log.error(f"Error exporting picture image for {base_name}-picture-{picture_counter}: {e}")

    def convert_pdfs_to_markdown(self) -> List[ConversionResult]:
        """
        Convert PDF files to Markdown using Docling.
        """
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        start = time.time()
        conv_results = list(converter.convert_all(self.input_paths, raises_on_error=False))
        self.conv_results = conv_results
        self.export_markdown_only(conv_results)
        self.export_images(conv_results)
        elapsed = time.time() - start
        _log.info(f"PDF to Markdown conversion completed in {elapsed:.2f} seconds.")
        return conv_results

    # === Part 2: Markdown Cleaning ===

    @staticmethod
    def clean_markdown(text: str, lowercase: bool = True, remove_stopwords: bool = True,
                       fix_spelling: bool = True, remove_unicode: bool = True, normalize: bool = True) -> str:
        """
        Clean Markdown text while preserving the <!--image--> marker.
        """
        text = re.sub(r"<!--\s*image\s*-->", "<!--image-->", text)
        placeholder_token = "XYZIMAGEXYZ"
        text = text.replace("<!--image-->", placeholder_token)
        html = markdown.markdown(text)
        plain_text = re.sub(r"<[^>]+>", "", html)
        if lowercase:
            plain_text = plain_text.lower()
        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            keep = {"not", "no", "nor", "none", "neither", "never", "nothing", "nobody", "nowhere"}
            filtered_stopwords = stop_words - keep
            plain_text = " ".join(word for word in plain_text.split() if word not in filtered_stopwords)
        if fix_spelling:
            spell = SpellChecker()
            corrected = []
            for word in plain_text.split():
                if word.isalpha() and not word.isupper():
                    corrected.append(spell.correction(word) or word)
                else:
                    corrected.append(word)
            plain_text = " ".join(corrected)
        if remove_unicode:
            plain_text = "".join(ch for ch in plain_text if ch.isascii())
        if normalize:
            contractions = {
                r"won't": "will not",
                r"can't": "cannot",
                r"n't": " not",
                r"'re": " are",
                r"'s": " is",
                r"'d": " would",
                r"'ll": " will",
                r"'ve": " have",
                r"'m": " am",
            }
            for pat, repl in contractions.items():
                plain_text = re.sub(pat, repl, plain_text)
            plain_text = re.sub(r"\bhrs\b", "hours", plain_text)
            plain_text = re.sub(r"\bmin\b", "minutes", plain_text)
            plain_text = re.sub(r"&", " and ", plain_text)
            plain_text = re.sub(r"w/", "with", plain_text)
            def num_to_word(match):
                num = int(match.group(1))
                return num2words(num) if num < 10 else match.group(1)
            plain_text = re.sub(r"\b(\d+)\b", num_to_word, plain_text)
            plain_text = re.sub(r"[^a-zA-Z0-9\s-]", "", plain_text)
            plain_text = re.sub(r"(?<!\w)-(?!\w)", " ", plain_text)
        plain_text = plain_text.replace(placeholder_token.lower(), "<!--image-->")
        return plain_text.strip()

    # === Part 3: Image Captioning and Markdown Post-Processing (Optional) ===

    @staticmethod
    def extract_image_metadata(conv_res: ConversionResult) -> List[Dict]:
        """
        Extract metadata for images from a conversion result.
        """
        image_meta = []
        from docling_core.types.doc import PictureItem, TableItem
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, (PictureItem, TableItem)):
                page_no = getattr(element, "page_no", "unknown")
                image_meta.append({"page_number": page_no})
        return image_meta

    @staticmethod
    def generate_caption(context: str, page_number, image_index: int) -> str:
        """
        Generate a stub caption based on context.
        
        """
        return f"Caption for image {image_index} on page {page_number}: {context[:50]}..."

    @staticmethod
    def insert_captions_into_markdown(markdown_text: str, image_metadata: List[Dict], context_window: int = 100) -> str:
        """
        Insert generated captions into the Markdown text at each <!--image--> placeholder.
        """
        markdown_text = re.sub(r"<!--\s*image\s*-->", "<!--image-->", markdown_text)
        segments = markdown_text.split("<!--image-->")
        new_text = segments[0]
        num_placeholders = len(segments) - 1
        for i in range(num_placeholders):
            before = segments[i][-context_window:] if len(segments[i]) > context_window else segments[i]
            after = segments[i+1][:context_window] if len(segments[i+1]) > context_window else segments[i+1]
            context = before + " " + after
            page_number = image_metadata[i].get("page_number", "unknown") if i < len(image_metadata) else "unknown"
            caption = DocumentIngestionPipeline.generate_caption(context, page_number, i)
            new_text += f"\n**Image Caption (Page {page_number}, Image {i}):** {caption}\n" + segments[i+1]
        return new_text

    # === Part 4: Chunking and Enrichment ===

    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 25) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    @staticmethod
    def enrich_chunk(chunk: str, source: str, chunk_index: int) -> Dict:
        """
        Enrich a text chunk with metadata.
        """
        return {
            "source": source,
            "chunk_index": chunk_index,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "length": len(chunk)
        }

    @staticmethod
    def chunks_to_langchain_docs(chunks: List[str], source: str) -> List[Document]:
        """
        Convert text chunks into LangChain Document objects with metadata.
        """
        docs = []
        for idx, chunk in enumerate(chunks):
            meta = DocumentIngestionPipeline.enrich_chunk(chunk, source, idx)
            docs.append(Document(page_content=chunk, metadata=meta))
        return docs

    # === Part 5: Integration – Create LangChain Documents ===

    def create_langchain_documents(self) -> List[Document]:
        """
        Run the full ingestion pipeline:
          1. Convert PDFs to Markdown.
          2. Load and (optionally) clean the Markdown.
          3. (Optionally) Insert image captions.
          4. Split the text into chunks.
          5. Convert chunks into LangChain Document objects.
        """
        self.convert_pdfs_to_markdown()
        # Assume processing the first input file's Markdown output.
        first_input = self.input_paths[0]
        md_path = self.output_dir / f"{first_input.stem}.md"
        self.markdown_file = md_path
        try:
            with md_path.open("r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            _log.error(f"Error reading Markdown file {md_path}: {e}")
            raise e
        if not md_text:
            raise ValueError("No Markdown text loaded from conversion.")

        # Process the Markdown content.
        processed_text = self.clean_markdown(md_text) if self.clean else md_text
        if self.caption:
            image_metadata = self.extract_image_metadata(self.conv_results[0])
            processed_text = self.insert_captions_into_markdown(processed_text, image_metadata, context_window=100)

        # Split and enrich the text.
        chunks = self.split_into_chunks(processed_text, chunk_size=400, overlap=50)
        docs = self.chunks_to_langchain_docs(chunks, source=md_path.name)
        return docs


# === Main Execution (for testing) ===

# if __name__ == "__main__":
#     # Adjust the PDF file paths as needed.
#     pdf_files = [Path("data/Report.pdf")]
#     md_output_dir = Path("converted_markdown")
#     pipeline = DocumentIngestionPipeline(input_paths=pdf_files, output_dir=md_output_dir, clean=True, caption=False)
#     langchain_docs = pipeline.create_langchain_documents()
#     _log.info(f"Created {len(langchain_docs)} LangChain Document objects.")
#     if langchain_docs:
#         print("First Document Content:")
#         print(langchain_docs[0].page_content)
#         print("First Document Metadata:")
#         print(langchain_docs[0].metadata)
