import type * as tiktoken from "js-tiktoken";
import { Document, BaseDocumentTransformer } from "@langchain/core/documents";
import { getEncoding } from "@langchain/core/utils/tiktoken";

type TextChunkContext = {
  chunk: string;
  chunkStartIndex: number;
  chunkEndIndex: number;
  chunkStartLine: number;
  chunkLineCount: number;
  textOrdinal: number;
  globalChunkOrdinal: number;
  textChunkOrdinal: number;
};

type UpdateMetadataFunction = (
  textMetadata: Record<string, any>,
  context: TextChunkContext
) => Record<string, any>;

export interface TextSplitterParams {
  chunkSize: number;
  chunkOverlap: number;
  keepSeparator: boolean;
  updateMetadataFunction?: UpdateMetadataFunction;
  lengthFunction?:
    | ((text: string) => number)
    | ((text: string) => Promise<number>);
}

export type TextSplitterChunkHeaderOptions = {
  chunkHeader?: string;
  chunkOverlapHeader?: string;
  appendChunkOverlapHeader?: boolean;
};

export abstract class TextSplitter
  extends BaseDocumentTransformer
  implements TextSplitterParams
{
  lc_namespace = ["langchain", "document_transformers", "text_splitters"];

  chunkSize = 1000;

  chunkOverlap = 200;

  keepSeparator = false;

  updateMetadataFunction: (
    documentMetadata: Record<string, any>,
    TextChunkContext: TextChunkContext
  ) => Record<string, any>;

  addLineNumbersToMetadata(
    documentMetadata: Record<string, any>,
    { chunkStartLine, chunkLineCount }: TextChunkContext
  ) {
    const updatedMetadata = documentMetadata;
    const loc = {
      lines: {
        from: chunkStartLine,
        to: chunkStartLine + chunkLineCount,
      },
    };
    updatedMetadata.loc =
      documentMetadata.loc && typeof documentMetadata.loc === "object"
        ? { ...documentMetadata.loc, ...loc.lines }
        : loc;
    return documentMetadata;
  }

  lengthFunction:
    | ((text: string) => number)
    | ((text: string) => Promise<number>);

  constructor(fields?: Partial<TextSplitterParams>) {
    super(fields);
    this.chunkSize = fields?.chunkSize ?? this.chunkSize;
    this.chunkOverlap = fields?.chunkOverlap ?? this.chunkOverlap;
    this.keepSeparator = fields?.keepSeparator ?? this.keepSeparator;
    this.updateMetadataFunction =
      fields?.updateMetadataFunction ?? this.addLineNumbersToMetadata;
    this.lengthFunction =
      fields?.lengthFunction ?? ((text: string) => text.length);
    if (this.chunkOverlap >= this.chunkSize) {
      throw new Error("Cannot have chunkOverlap >= chunkSize");
    }
  }

  async transformDocuments(
    documents: Document[],
    chunkHeaderOptions: TextSplitterChunkHeaderOptions = {}
  ): Promise<Document[]> {
    return this.splitDocuments(documents, chunkHeaderOptions);
  }

  abstract splitText(text: string): Promise<string[]>;

  protected splitOnSeparator(text: string, separator: string): string[] {
    let splits;
    if (separator) {
      if (this.keepSeparator) {
        const regexEscapedSeparator = separator.replace(
          /[/\-\\^$*+?.()|[\]{}]/g,
          "\\$&"
        );
        splits = text.split(new RegExp(`(?=${regexEscapedSeparator})`));
      } else {
        splits = text.split(separator);
      }
    } else {
      splits = text.split("");
    }
    return splits.filter((s) => s !== "");
  }

  async createDocuments(
    texts: string[],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    metadatas: Record<string, any>[] = [],
    chunkHeaderOptions: TextSplitterChunkHeaderOptions = {}
  ): Promise<Document[]> {
    // if no metadata is provided, we create an empty one for each text
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const _metadatas: Record<string, any>[] =
      metadatas.length > 0
        ? metadatas
        : [...Array(texts.length)].map(() => ({}));
    const {
      chunkHeader = "",
      chunkOverlapHeader = "(cont'd) ",
      appendChunkOverlapHeader = false,
    } = chunkHeaderOptions;
    const documents = new Array<Document>();
    for (let i = 0; i < texts.length; i += 1) {
      const text = texts[i];
      let lineCounterIndex = 1;
      let prevChunk = null;
      let indexPrevChunk = -1;
      const chunks = await this.splitText(text);
      for (let j = 0; j < chunks.length; j += 1) {
        const chunk = chunks[j];
        let pageContent = chunkHeader;

        // we need to count the \n that are in the text before getting removed by the splitting
        const indexChunk = text.indexOf(chunk, indexPrevChunk + 1);
        if (prevChunk === null) {
          const newLinesBeforeFirstChunk = this.numberOfNewLines(
            text,
            0,
            indexChunk
          );
          lineCounterIndex += newLinesBeforeFirstChunk;
        } else {
          const indexEndPrevChunk =
            indexPrevChunk + (await this.lengthFunction(prevChunk));
          if (indexEndPrevChunk < indexChunk) {
            const numberOfIntermediateNewLines = this.numberOfNewLines(
              text,
              indexEndPrevChunk,
              indexChunk
            );
            lineCounterIndex += numberOfIntermediateNewLines;
          } else if (indexEndPrevChunk > indexChunk) {
            const numberOfIntermediateNewLines = this.numberOfNewLines(
              text,
              indexChunk,
              indexEndPrevChunk
            );
            lineCounterIndex -= numberOfIntermediateNewLines;
          }
          if (appendChunkOverlapHeader) {
            pageContent += chunkOverlapHeader;
          }
        }
        const newLinesCount = this.numberOfNewLines(chunk);

        const updatedMetadata = this.updateMetadataFunction(
          { ..._metadatas[i] },
          {
            chunk,
            chunkStartIndex: indexChunk,
            chunkEndIndex: indexChunk + (await this.lengthFunction(chunk)),
            chunkStartLine: lineCounterIndex,
            chunkLineCount: newLinesCount,
            textOrdinal: i + 1,
            globalChunkOrdinal: documents.length,
            textChunkOrdinal: j + 1,
          }
        );

        pageContent += chunk;
        documents.push(
          new Document({ pageContent, metadata: updatedMetadata })
        );
        lineCounterIndex += newLinesCount;
        prevChunk = chunk;
        indexPrevChunk = indexChunk;
      }
    }
    return documents;
  }

  private numberOfNewLines(text: string, start?: number, end?: number) {
    const textSection = text.slice(start, end);
    return (textSection.match(/\n/g) || []).length;
  }

  async splitDocuments(
    documents: Document[],
    chunkHeaderOptions: TextSplitterChunkHeaderOptions = {}
  ): Promise<Document[]> {
    const selectedDocuments = documents.filter(
      (doc) => doc.pageContent !== undefined
    );
    const texts = selectedDocuments.map((doc) => doc.pageContent);
    const metadatas = selectedDocuments.map((doc) => doc.metadata);
    return this.createDocuments(texts, metadatas, chunkHeaderOptions);
  }

  private joinDocs(docs: string[], separator: string): string | null {
    const text = docs.join(separator).trim();
    return text === "" ? null : text;
  }

  async mergeSplits(splits: string[], separator: string): Promise<string[]> {
    const docs: string[] = [];
    const currentDoc: string[] = [];
    let total = 0;
    for (const d of splits) {
      const _len = await this.lengthFunction(d);
      if (
        total + _len + (currentDoc.length > 0 ? separator.length : 0) >
        this.chunkSize
      ) {
        if (total > this.chunkSize) {
          console.warn(
            `Created a chunk of size ${total}, +
which is longer than the specified ${this.chunkSize}`
          );
        }
        if (currentDoc.length > 0) {
          const doc = this.joinDocs(currentDoc, separator);
          if (doc !== null) {
            docs.push(doc);
          }
          // Keep on popping if:
          // - we have a larger chunk than in the chunk overlap
          // - or if we still have any chunks and the length is long
          while (
            total > this.chunkOverlap ||
            (total + _len > this.chunkSize && total > 0)
          ) {
            total -= await this.lengthFunction(currentDoc[0]);
            currentDoc.shift();
          }
        }
      }
      currentDoc.push(d);
      total += _len;
    }
    const doc = this.joinDocs(currentDoc, separator);
    if (doc !== null) {
      docs.push(doc);
    }
    return docs;
  }
}

export interface CharacterTextSplitterParams extends TextSplitterParams {
  separator: string;
}

export class CharacterTextSplitter
  extends TextSplitter
  implements CharacterTextSplitterParams
{
  static lc_name() {
    return "CharacterTextSplitter";
  }

  separator = "\n\n";

  constructor(fields?: Partial<CharacterTextSplitterParams>) {
    super(fields);
    this.separator = fields?.separator ?? this.separator;
  }

  async splitText(text: string): Promise<string[]> {
    // First we naively split the large input into a bunch of smaller ones.
    const splits = this.splitOnSeparator(text, this.separator);
    return this.mergeSplits(splits, this.keepSeparator ? "" : this.separator);
  }
}

export interface RecursiveCharacterTextSplitterParams
  extends TextSplitterParams {
  separators: string[];
}

export const SupportedTextSplitterLanguages = [
  "cpp",
  "go",
  "java",
  "js",
  "php",
  "proto",
  "python",
  "rst",
  "ruby",
  "rust",
  "scala",
  "swift",
  "markdown",
  "latex",
  "html",
  "sol",
] as const;

export type SupportedTextSplitterLanguage =
  (typeof SupportedTextSplitterLanguages)[number];

export class RecursiveCharacterTextSplitter
  extends TextSplitter
  implements RecursiveCharacterTextSplitterParams
{
  static lc_name() {
    return "RecursiveCharacterTextSplitter";
  }

  separators: string[] = ["\n\n", "\n", " ", ""];

  constructor(fields?: Partial<RecursiveCharacterTextSplitterParams>) {
    super(fields);
    this.separators = fields?.separators ?? this.separators;
    this.keepSeparator = fields?.keepSeparator ?? true;
  }

  private async _splitText(text: string, separators: string[]) {
    const finalChunks: string[] = [];

    // Get appropriate separator to use
    let separator: string = separators[separators.length - 1];
    let newSeparators;
    for (let i = 0; i < separators.length; i += 1) {
      const s = separators[i];
      if (s === "") {
        separator = s;
        break;
      }
      if (text.includes(s)) {
        separator = s;
        newSeparators = separators.slice(i + 1);
        break;
      }
    }

    // Now that we have the separator, split the text
    const splits = this.splitOnSeparator(text, separator);

    // Now go merging things, recursively splitting longer texts.
    let goodSplits: string[] = [];
    const _separator = this.keepSeparator ? "" : separator;
    for (const s of splits) {
      if ((await this.lengthFunction(s)) < this.chunkSize) {
        goodSplits.push(s);
      } else {
        if (goodSplits.length) {
          const mergedText = await this.mergeSplits(goodSplits, _separator);
          finalChunks.push(...mergedText);
          goodSplits = [];
        }
        if (!newSeparators) {
          finalChunks.push(s);
        } else {
          const otherInfo = await this._splitText(s, newSeparators);
          finalChunks.push(...otherInfo);
        }
      }
    }
    if (goodSplits.length) {
      const mergedText = await this.mergeSplits(goodSplits, _separator);
      finalChunks.push(...mergedText);
    }
    return finalChunks;
  }

  async splitText(text: string): Promise<string[]> {
    return this._splitText(text, this.separators);
  }

  static fromLanguage(
    language: SupportedTextSplitterLanguage,
    options?: Partial<RecursiveCharacterTextSplitterParams>
  ) {
    return new RecursiveCharacterTextSplitter({
      ...options,
      separators:
        RecursiveCharacterTextSplitter.getSeparatorsForLanguage(language),
    });
  }

  static getSeparatorsForLanguage(language: SupportedTextSplitterLanguage) {
    if (language === "cpp") {
      return [
        // Split along class definitions
        "\nclass ",
        // Split along function definitions
        "\nvoid ",
        "\nint ",
        "\nfloat ",
        "\ndouble ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "go") {
      return [
        // Split along function definitions
        "\nfunc ",
        "\nvar ",
        "\nconst ",
        "\ntype ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nswitch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "java") {
      return [
        // Split along class definitions
        "\nclass ",
        // Split along method definitions
        "\npublic ",
        "\nprotected ",
        "\nprivate ",
        "\nstatic ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "js") {
      return [
        // Split along function definitions
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nclass ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "php") {
      return [
        // Split along function definitions
        "\nfunction ",
        // Split along class definitions
        "\nclass ",
        // Split along control flow statements
        "\nif ",
        "\nforeach ",
        "\nwhile ",
        "\ndo ",
        "\nswitch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "proto") {
      return [
        // Split along message definitions
        "\nmessage ",
        // Split along service definitions
        "\nservice ",
        // Split along enum definitions
        "\nenum ",
        // Split along option definitions
        "\noption ",
        // Split along import statements
        "\nimport ",
        // Split along syntax declarations
        "\nsyntax ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "python") {
      return [
        // First, try to split along class definitions
        "\nclass ",
        "\ndef ",
        "\n\tdef ",
        // Now split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "rst") {
      return [
        // Split along section titles
        "\n===\n",
        "\n---\n",
        "\n***\n",
        // Split along directive markers
        "\n.. ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "ruby") {
      return [
        // Split along method definitions
        "\ndef ",
        "\nclass ",
        // Split along control flow statements
        "\nif ",
        "\nunless ",
        "\nwhile ",
        "\nfor ",
        "\ndo ",
        "\nbegin ",
        "\nrescue ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "rust") {
      return [
        // Split along function definitions
        "\nfn ",
        "\nconst ",
        "\nlet ",
        // Split along control flow statements
        "\nif ",
        "\nwhile ",
        "\nfor ",
        "\nloop ",
        "\nmatch ",
        "\nconst ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "scala") {
      return [
        // Split along class definitions
        "\nclass ",
        "\nobject ",
        // Split along method definitions
        "\ndef ",
        "\nval ",
        "\nvar ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nmatch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "swift") {
      return [
        // Split along function definitions
        "\nfunc ",
        // Split along class definitions
        "\nclass ",
        "\nstruct ",
        "\nenum ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\ndo ",
        "\nswitch ",
        "\ncase ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "markdown") {
      return [
        // First, try to split along Markdown headings (starting with level 2)
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n##### ",
        "\n###### ",
        // Note the alternative syntax for headings (below) is not handled here
        // Heading level 2
        // ---------------
        // End of code block
        "```\n\n",
        // Horizontal lines
        "\n\n***\n\n",
        "\n\n---\n\n",
        "\n\n___\n\n",
        // Note that this splitter doesn't handle horizontal lines defined
        // by *three or more* of ***, ---, or ___, but this is not handled
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "latex") {
      return [
        // First, try to split along Latex sections
        "\n\\chapter{",
        "\n\\section{",
        "\n\\subsection{",
        "\n\\subsubsection{",

        // Now split by environments
        "\n\\begin{enumerate}",
        "\n\\begin{itemize}",
        "\n\\begin{description}",
        "\n\\begin{list}",
        "\n\\begin{quote}",
        "\n\\begin{quotation}",
        "\n\\begin{verse}",
        "\n\\begin{verbatim}",

        // Now split by math environments
        "\n\\begin{align}",
        "$$",
        "$",

        // Now split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else if (language === "html") {
      return [
        // First, try to split along HTML tags
        "<body>",
        "<div>",
        "<p>",
        "<br>",
        "<li>",
        "<h1>",
        "<h2>",
        "<h3>",
        "<h4>",
        "<h5>",
        "<h6>",
        "<span>",
        "<table>",
        "<tr>",
        "<td>",
        "<th>",
        "<ul>",
        "<ol>",
        "<header>",
        "<footer>",
        "<nav>",
        // Head
        "<head>",
        "<style>",
        "<script>",
        "<meta>",
        "<title>",
        // Normal type of lines
        " ",
        "",
      ];
    } else if (language === "sol") {
      return [
        // Split along compiler informations definitions
        "\npragma ",
        "\nusing ",
        // Split along contract definitions
        "\ncontract ",
        "\ninterface ",
        "\nlibrary ",
        // Split along method definitions
        "\nconstructor ",
        "\ntype ",
        "\nfunction ",
        "\nevent ",
        "\nmodifier ",
        "\nerror ",
        "\nstruct ",
        "\nenum ",
        // Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\ndo while ",
        "\nassembly ",
        // Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
      ];
    } else {
      throw new Error(`Language ${language} is not supported.`);
    }
  }
}

export interface TokenTextSplitterParams extends TextSplitterParams {
  encodingName: tiktoken.TiktokenEncoding;
  allowedSpecial: "all" | Array<string>;
  disallowedSpecial: "all" | Array<string>;
}

/**
 * Implementation of splitter which looks at tokens.
 */
export class TokenTextSplitter
  extends TextSplitter
  implements TokenTextSplitterParams
{
  static lc_name() {
    return "TokenTextSplitter";
  }

  encodingName: tiktoken.TiktokenEncoding;

  allowedSpecial: "all" | Array<string>;

  disallowedSpecial: "all" | Array<string>;

  private tokenizer: tiktoken.Tiktoken;

  constructor(fields?: Partial<TokenTextSplitterParams>) {
    super(fields);

    this.encodingName = fields?.encodingName ?? "gpt2";
    this.allowedSpecial = fields?.allowedSpecial ?? [];
    this.disallowedSpecial = fields?.disallowedSpecial ?? "all";
  }

  async splitText(text: string): Promise<string[]> {
    if (!this.tokenizer) {
      this.tokenizer = await getEncoding(this.encodingName);
    }

    const splits: string[] = [];

    const input_ids = this.tokenizer.encode(
      text,
      this.allowedSpecial,
      this.disallowedSpecial
    );

    let start_idx = 0;

    while (start_idx < input_ids.length) {
      if (start_idx > 0) {
        start_idx -= this.chunkOverlap;
      }
      const end_idx = Math.min(start_idx + this.chunkSize, input_ids.length);
      const chunk_ids = input_ids.slice(start_idx, end_idx);
      splits.push(this.tokenizer.decode(chunk_ids));
      start_idx = end_idx;
    }

    return splits;
  }
}

export type MarkdownTextSplitterParams = TextSplitterParams;

export class MarkdownTextSplitter
  extends RecursiveCharacterTextSplitter
  implements MarkdownTextSplitterParams
{
  constructor(fields?: Partial<MarkdownTextSplitterParams>) {
    super({
      ...fields,
      separators:
        RecursiveCharacterTextSplitter.getSeparatorsForLanguage("markdown"),
    });
  }
}

export type LatexTextSplitterParams = TextSplitterParams;

export class LatexTextSplitter
  extends RecursiveCharacterTextSplitter
  implements LatexTextSplitterParams
{
  constructor(fields?: Partial<LatexTextSplitterParams>) {
    super({
      ...fields,
      separators:
        RecursiveCharacterTextSplitter.getSeparatorsForLanguage("latex"),
    });
  }
}
