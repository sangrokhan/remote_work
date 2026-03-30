# PDF Roundtrip Extraction PRD

## Purpose

This document is a working guide for agents building a PDF parser for PDFs generated from Word documents.

## Input Document Variability

The input PDF is produced by combining multiple functional specification documents written by different authors.

At a high level, these documents follow a common writing format.

However, individual authors sometimes change Word settings related to table output, and in some cases they also use fonts in ways that do not follow the expected constraints.

Because of this, the resulting PDF may contain inconsistencies in table structure, layout behavior, and font usage even when the documents belong to the same overall documentation set.

## Extraction Targets

The parser should handle the following extraction targets.

### Text

The document contains section headers and the corresponding body content.

Except for chapter headings, the remaining headers are left-aligned.

The body area is written with a fixed indent from the left side.

### Tables

Tables occupy the full available width within the page's left and right content boundaries.

Their header area is filled with a specific color.

The outer left and right sides of the table do not have vertical border lines.

Table borders and cell boundaries are drawn with black lines.

Merged cells are used, and in some cases the merge relationship continues across page boundaries.

Some tables repeat their header on the next page, while others do not.

### Note Areas

There are note areas that are similar to tables.

Unlike tables, note areas use blue borders.

Note areas may contain multiple notes inside a single note group.

A note group is identified by blue horizontal lines that define the note-group region.

They occupy the width of the text area rather than the full table-width region.

They also follow the same indent depth used by the text body.

A note area starts with a note image.

The note image must be excluded from the output.

### Images

There are image regions surrounded by a gray outer border.

These regions may contain JPG or PNG images.

They may also contain content created with Word or PowerPoint features, such as sequence charts.

Because of this, text-like elements may appear inside an image region.

Any text found inside the gray-bordered image area must be treated as part of the image rather than as normal text.

That text must not be included in text extraction output.

The content of the image region should instead be reconstructed and output as an image.

## Output and Naming Requirements

### Text and Note Markdown

Each technical document must produce its own main Markdown file.

The main Markdown file name must use the format `doc-id.md`.

If no `h2` has been identified yet, the default file name must use `output.md`.

Text content and note content must be written into the same main Markdown file as part of the same document flow.

### Table Markdown

Each technical document must produce one table Markdown file.

The table Markdown file name must use the format `doc-id_tables.md`.

Table numbering must restart for each technical document.

Table numbering must be distinguishable by including the technical document identity, the table label, and the table number.

The main Markdown file must contain a plain-text reference to the table output at the original table position.

The table reference format must be `[FGR-BC0401_tables.md - Table 1]`.

This reference must be written as plain text and must not be generated as a Markdown link.

### Image Output

Each technical document must produce its own image directory.

The image directory name must use the format `doc-id_images`.

Image numbering must restart for each technical document.

If the source image is a JPG or PNG image, it must be output in that image format.

If the image region corresponds to content created with Word or PowerPoint features, it must be output as a PNG image.

Text found inside a gray-bordered image region must be treated as part of the image and must not be included in normal text extraction.

The main Markdown file must contain a plain-text reference to the image output at the original image position.

The image reference format must be `[FGR-BC0401_image_1.png]`.

This reference must be written as plain text and must not be generated as a Markdown link.

### Debug Outputs

Debug-only outputs may include page markers for inspection purposes.

When debug page markers are enabled, the parser may insert page comments such as `[//]: # (Page N)` into the generated Markdown outputs.

These page comments are debug metadata and are not part of the primary extraction contract.

### Heading Rules

Headers must be rendered in Markdown using heading markers.

A separate JSON file will provide the font sizes used to classify headings.

The font size defined for chapter must be mapped to `h1`.

The next three font sizes in the JSON must be mapped to `h2`, `h3`, and `h4`.

Font sizes not defined in the JSON must be treated as normal body text.

The `h2` heading represents the technical document name.

The first 10 characters of the `h2` text must be used as the document identifier.

The document identifier is used to determine the output file names and image directory name.

## Header and Footer Constraints

Headers and footers are present on all pages.

They are not part of the parsing target and should be removed.

An exception exists on pages that contain chapter content.

On those pages, the header is absent.

Headers and footers are visually separated from the main body by straight-line elements placed at the top and bottom of the body region.

Because of this, they should be relatively easy to distinguish from the main content area.

An image is also present in the header region.

That header image is not part of the parsing target and must be removed.

## Watermark Constraints

A text-based watermark is present on all pages.

The watermark is not part of the parsing target and must be removed after parsing.

The watermark may overlap body text, tables, and image regions.

It must be removed as a text element rather than by masking or removing the full overlapped area.

The parser should remove only the watermark itself so that the underlying body, table, or image content remains available in the output.

## Document Range and Output Rules

A technical document begins at the point where an `h2` heading appears.

Its range continues until the next `h2` heading appears.

If chapter content appears between technical documents, it must not be included in the previous technical document.

Instead, that chapter content must be placed at the beginning of the next technical document output.

Outputs should be written under the `artifacts` directory.

Under `artifacts`, outputs should be organized in a way that follows the technical document naming rule.

## Parsing Order

The parser should follow a structure-first parsing order.

1. Detect and remove header and footer regions from the page.
2. Detect heading candidates using the provided font-size JSON rules.
3. Classify chapter and `h2` boundaries.
4. Determine the active technical document range.
5. Detect note regions.
6. Detect table regions.
7. Detect image regions.
8. Exclude text that belongs to detected table, note marker, image, header, and footer regions from normal body-text extraction.
9. Extract the remaining body text.
10. Reconstruct note text from detected note regions.
11. Merge tables that continue across page boundaries.
12. Insert plain-text references for table and image outputs at their original positions in the main Markdown flow.
13. Write outputs by technical document under the `artifacts` directory.

## Table Markdown Shape

Each technical document must write all tables into a single `doc-id_tables.md` file.

Each table must be identified with a comment-style title that includes the technical document identity and the table number.

The table content itself must be written using standard Markdown table syntax.

A table that continues across page boundaries must be merged and written as a single final table.

If a repeated table header appears during page continuation, the duplicated header row must be removed.

Merged cells must be represented without text repetition, leaving the continued cell positions empty where needed.

If body content or note content appears between two table fragments before table merging, those fragments must not be treated as one table.

That rule applies even when the table layout looks similar across the page boundary.

## Text Reconstruction Rules

Body text should be reconstructed only after note regions, table regions, image regions, header regions, footer regions, and watermark elements have been excluded from normal text extraction.

For normal body text, lines that belong to the same sentence should be merged into a single line in the Markdown output.

The fixed left indent of the body area should be treated as the baseline text start position for normal paragraph reconstruction.

Headers and body text should be reconstructed according to the heading rules defined by the external font-size JSON.

If a font size is mapped in the JSON, the corresponding Markdown heading marker must be applied.

If a font size is not mapped in the JSON, the text must be treated as normal body text.

Note content must be reconstructed as part of the same Markdown flow as body text.

Each note begins when a note image appears inside a blue-bordered note group.

The note image itself must not be output.

Each note must begin with the `Note:` label.

A single note group may contain multiple notes.

If multiple lines belong to the same note sentence, they must be merged into a single line.

If another note image appears, it must be treated as the start of a new note, and the next note must begin on a new line.

## Exclusion Rules

The following elements are not part of the output target and must be excluded from parsing results:

- header text
- footer text
- the header-region image
- watermark text
- note-start images
- text contained inside gray-bordered image regions

These elements must not appear in the main Markdown output.

They must also not be treated as normal body text, note text, or table content.

## Validation Rules

The parser output is considered valid when the following conditions are satisfied:

- header, footer, header image, and watermark content do not appear in the outputs
- each technical document is split into its own main Markdown file
- each technical document has its own table Markdown file and image directory
- `h2` is used to determine the technical document boundary and document identifier
- chapter content is placed at the beginning of the next technical document output
- tables that continue across pages are merged into one table when no body or note content exists between them
- repeated table headers are removed during table merging
- note content is included in the main Markdown flow with the `Note:` label
- text inside gray-bordered image regions is excluded from text output
- table and image references appear in the main Markdown file at their original positions

## Primary Objective

Build a repeatable PDF parser workflow that recovers content from Word-generated PDFs by using data exposed by PDF parsing libraries.

The parser should prioritize:

- text objects
- character and line coordinates
- page layout information
- detected table boundaries
- embedded image metadata

OCR is out of scope for this project.

Agents must not introduce OCR as a default path, optional path, or fallback path.

## Agent Guidance

Agents should assume that this project is about implementing a structured PDF parser, not manually extracting content from individual files and not using image-first recognition.

Agents should design the parser so that it can extract and reconstruct:

- body text
- tables
- embedded images
- practical reading order

Agents should focus on parser behavior, reconstruction rules, and output shape.

Agents should preserve document usability over perfect visual reproduction.

Agents should produce parser outputs that are easy to inspect, validate, and refine in later steps.

## Explicit Constraints

- Do not use OCR.
- Do not design fallback logic around OCR.
- Do not frame image-based recognition as part of the implementation path.
- Keep the solution strictly within PDF-library-based extraction and reconstruction.

## Success Condition

The workflow is successful when the implemented parser can extract meaningful content from a Word-generated PDF while keeping the process repeatable, debuggable, and suitable for later verification.
