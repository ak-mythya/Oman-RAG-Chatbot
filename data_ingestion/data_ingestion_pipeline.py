#!/usr/bin/env python3
"""
Document Ingestion Pipeline – Markdown Only

This class implements a pipeline that:
  1. Converts PDF files to Markdown using Docling.
  2. (Optionally) Inserts generated image captions into the Markdown.
  3. Splits the processed Markdown into chunks (semantic or fixed-size).
  4. Enriches each chunk with metadata and converts them into LangChain Document objects.
  5. Saves the LangChain Document objects to a JSON file.
"""

import os
import re
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Dict, Union

# --- Docling Imports ---
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions

# --- LangChain Imports ---
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import embeddings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    def __init__(self, input_paths: Union[str, Path, List[Union[str, Path]]], output_dir: Union[str, Path], clean: bool = False, caption: bool = False, embeddings=None):
        # Handle input_paths: convert to list of Path objects
        if isinstance(input_paths, (str, Path)):
            input_paths = [Path(input_paths)]  # Automatically convert string to Path
        elif isinstance(input_paths, list):
            input_paths = [Path(p) if isinstance(p, str) else p for p in input_paths]
        else:
            raise ValueError("input_paths must be a string, Path, or list of strings/Paths")

        pdf_files = []
        for path in input_paths:
            if path.is_dir():
                pdfs = [p for p in path.glob("*.pdf") if p.is_file()]
                if not pdfs:
                    _log.warning(f"No PDF files found in directory: {path}")
                pdf_files.extend(pdfs)
            elif path.is_file() and path.suffix.lower() == '.pdf':
                pdf_files.append(path)
            else:
                raise ValueError(f"Invalid path: {path}. Must be a PDF file or a directory containing PDFs.")

        if not pdf_files:
            raise ValueError("No valid PDF files found in the provided input.")

        self.input_paths = pdf_files
        self.output_dir = Path(output_dir)  # Automatically convert output_dir to Path
        self.clean = clean
        self.caption = caption
        self.conv_results: List[ConversionResult] = []
        self.markdown_file: Path = None
        _log.info(f"Found {len(self.input_paths)} PDF files to process.")

    def export_markdown_only(self, conv_results: Iterable[ConversionResult]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                base_name = conv_res.input.file.stem
                md_file = self.output_dir / f"{base_name}.md"
                md_file.write_text(conv_res.document.export_to_markdown(), encoding="utf-8")
                _log.info(f"Exported Markdown: {md_file}")
            else:
                _log.warning(f"Conversion failed or partial for: {conv_res.input.file}")

    def export_images(self, conv_results: Iterable[ConversionResult]) -> None:
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
        # --- OCR configuration for Arabic ---
        ocr_options = TesseractCliOcrOptions()
        ocr_options.lang = ["ara", "eng"]  # Arabic and English language codes
        ocr_options.force_full_page_ocr = True  # Forces OCR on all pages

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.images_scale = 2.0  # Reduced to minimize memory usage
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

    @staticmethod
    def extract_image_metadata(conv_res: ConversionResult) -> List[Dict]:
        image_meta = []
        from docling_core.types.doc import PictureItem, TableItem
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, (PictureItem, TableItem)):
                page_no = getattr(element, "page_no", "unknown")
                image_meta.append({"page_number": page_no})
        return image_meta

    @staticmethod
    def generate_caption(context: str, page_number, image_index: int) -> str:
        return f"Caption for image {image_index} on page {page_number}: {context[:50]}..."

    @staticmethod
    def insert_captions_into_markdown(markdown_text: str, image_metadata: List[Dict], context_window: int = 100) -> str:
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

    def create_langchain_documents(self, use_semantic_chunking: bool = False) -> List[Document]:
        self.convert_pdfs_to_markdown()
        documents = []

        for input_path in self.input_paths:
            md_path = self.output_dir / f"{input_path.stem}.md"
            self.markdown_file = md_path
            try:
                with md_path.open("r", encoding="utf-8") as f:
                    md_text = f.read()
            except FileNotFoundError:
                _log.error(f"Markdown file {md_path} not found, likely due to conversion failure.")
                continue
            if not md_text:
                _log.warning(f"Markdown file {md_path} is empty.")
                continue

            processed_text = md_text
            if self.caption:
                # Find the conversion result for this input file
                conv_res = next((cr for cr in self.conv_results if cr.input.file == input_path), None)
                if conv_res:
                    image_metadata = self.extract_image_metadata(conv_res)
                    processed_text = self.insert_captions_into_markdown(processed_text, image_metadata, context_window=100)

            if embeddings is None:
                raise ValueError("Embeddings must be provided for semantic chunking.")

            base_metadata = {"source": md_path.name}
            sentence_split_regex = r'(?<=[.؟!؛])\s+'
            semantic_chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                sentence_split_regex=sentence_split_regex,
                breakpoint_threshold_amount=65,
                min_chunk_size=3
            )

            docs = semantic_chunker.create_documents([processed_text], metadatas=[base_metadata])
            documents.extend(docs)

        return documents

if __name__ == "__main__":
    # Example usage with both input and output as strings
    pdf_dir = "/kaggle/input/arabic-data"  # String path to directory
    md_output_dir = "/kaggle/working"  # String path to output directory

    print("Testing semantic chunking...")
    pipeline_semantic = DocumentIngestionPipeline(
        input_paths=pdf_dir,  # Pass the input string directly
        output_dir=md_output_dir,  # Pass the output string directly
        clean=False,
        caption=False,
        embeddings=embeddings
    )
    langchain_docs_semantic = pipeline_semantic.create_langchain_documents(use_semantic_chunking=True)
    print("Langchain_semantics_chunks", langchain_docs_semantic)
    print(f"Created {len(langchain_docs_semantic)} LangChain Document objects with semantic chunking.")
    if langchain_docs_semantic:
        print("First Document Content (Semantic):")
        print(langchain_docs_semantic[0].page_content[:200])
        print("First Document Metadata (Semantic):")
        print(langchain_docs_semantic[0].metadata)