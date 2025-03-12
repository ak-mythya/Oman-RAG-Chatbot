import logging
from pathlib import Path
from oman_chatbot_new.data_ingestion.data_ingestion_pipeline import DocumentIngestionPipeline  # Adjust the import path as needed

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Provide the path to your sample PDF file.
    pdf_file = Path("oman_chatbot_new/data_ingestion/data/Report.pdf")
    # Specify an output directory (this can be a temporary or dedicated directory)
    output_dir = Path("oman_chatbot_new/testing/data_ingestion/pipeline_output")
    
    # Initialize the pipeline with desired options.
    pipeline = DocumentIngestionPipeline(
        input_paths=[pdf_file],
        output_dir=output_dir,
        clean=True,      # set to True to clean the Markdown text
        caption=True     # set to True to insert image captions
    )
    
    # Step 1: Convert PDF to Markdown (and export images, if any)
    logging.info("Starting PDF to Markdown conversion...")
    conversion_results = pipeline.convert_pdfs_to_markdown()
    logging.info(f"Conversion output saved in: {output_dir}")

    # Step 2: Load the generated Markdown file.
    md_path = output_dir / f"{pdf_file.stem}.md"
    try:
        with md_path.open("r", encoding="utf-8") as f:
            markdown_text = f.read()
        logging.info(f"Loaded Markdown from {md_path}")
    except Exception as e:
        logging.error(f"Failed to read Markdown file: {e}")
        return

    # Step 3: Clean the Markdown (if enabled) and save cleaned output.
    if pipeline.clean:
        cleaned_text = pipeline.clean_markdown(markdown_text)
        cleaned_md_path = output_dir / f"{pdf_file.stem}_cleaned.md"
        cleaned_md_path.write_text(cleaned_text, encoding="utf-8")
        logging.info(f"Cleaned Markdown saved to: {cleaned_md_path}")
    else:
        cleaned_text = markdown_text

    # Step 4: Insert image captions (if enabled) and save the result.
    if pipeline.caption:
        # We assume the conversion produced at least one conversion result.
        image_metadata = pipeline.extract_image_metadata(conversion_results[0])
        captioned_text = pipeline.insert_captions_into_markdown(cleaned_text, image_metadata, context_window=100)
        captioned_md_path = output_dir / f"{pdf_file.stem}_captioned.md"
        captioned_md_path.write_text(captioned_text, encoding="utf-8")
        logging.info(f"Captioned Markdown saved to: {captioned_md_path}")
        # Use captioned_text for further processing.
        processed_text = captioned_text
    else:
        processed_text = cleaned_text

    # Step 5: Split the processed Markdown into chunks.
    chunks = pipeline.split_into_chunks(processed_text, chunk_size=400, overlap=50)
    logging.info(f"Split text into {len(chunks)} chunks.")

    # Optional: Convert chunks into LangChain Document objects.
    docs = pipeline.chunks_to_langchain_docs(chunks, source=md_path.name)

    # Save a summary file with chunk metadata and a preview of each chunk.
    chunks_output_path = output_dir / f"{pdf_file.stem}_chunks.txt"
    with chunks_output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(f"Chunk Index: {doc.metadata['chunk_index']}\n")
            f.write(f"Timestamp: {doc.metadata['timestamp']}\n")
            f.write(f"Length: {doc.metadata['length']} characters\n")
            f.write("Content Preview:\n")
            f.write(doc.page_content[:200] + "\n")  # save first 200 characters as a preview
            f.write("\n" + "-" * 50 + "\n")
    logging.info(f"Chunk summaries saved to: {chunks_output_path}")

if __name__ == "__main__":
    main()
