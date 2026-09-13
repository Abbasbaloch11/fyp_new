import os
import re
import json
import pandas as pd

from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. Please add it to your .env file."
    )


# =========================================================
# RAG SEARCH
# =========================================================

class RAGSearch:

    def __init__(
        self,
        persist_dir="faiss_store",
        data_dir="data",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gemini-2.5-flash"
    ):

        self.data_dir = data_dir

        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        # -------------------------------------------------
        # Load FAISS
        # -------------------------------------------------

        if os.path.exists(faiss_path) and os.path.exists(meta_path):

            self.vectorstore.load()

        else:

            print(
                "[INFO] FAISS index not found. Building new index..."
            )

            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)


        # -------------------------------------------------
        # Load CSV data
        # -------------------------------------------------

        self.dataframe = self.load_student_data()

        # -------------------------------------------------
        # Gemini
        # -------------------------------------------------

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=GOOGLE_API_KEY
        )

        print(
            f"[INFO] Gemini LLM initialized: {llm_model}"
        )


    # =====================================================
    # LOAD CSV
    # =====================================================

    def load_student_data(self):
        data_dir = self.data_dir

        if not os.path.isdir(data_dir):
            print(f"[WARNING] Data folder not found: {data_dir}")
            return None

        csv_files = [
            file
            for file in os.listdir(data_dir)
            if file.lower().endswith(".csv")
        ]

        if not csv_files:

            print("[WARNING] No CSV file found.")

            return None

        csv_path = os.path.join(
            data_dir,
            csv_files[0]
        )

        print(
            f"[INFO] Loading structured student data: {csv_path}"
        )

        df = pd.read_csv(csv_path)

        print(
            f"[INFO] Loaded {len(df)} structured student records."
        )

        print(
            f"[INFO] Columns: {list(df.columns)}"
        )

        return df


    # =====================================================
    # NORMALIZE TEXT
    # =====================================================

    def normalize(self, value):

        if value is None:
            return ""

        return str(value).strip().lower()


    # =====================================================
    # GET STUDENT ID
    # =====================================================

    def extract_student_id(self, query):

        pattern = r"\bSTU\d+\b"

        match = re.search(
            pattern,
            query.upper()
        )

        if match:

            return match.group(0)

        return None


    # =====================================================
    # FIND COLUMN
    # =====================================================

    def find_column(self, keywords):

        if self.dataframe is None:
            return None

        for column in self.dataframe.columns:

            column_lower = column.lower()

            for keyword in keywords:

                if keyword.lower() in column_lower:

                    return column

        return None


    # =====================================================
    # EXACT STUDENT ID SEARCH
    # =====================================================

    def search_by_student_id(self, student_id):

        if self.dataframe is None:
            return pd.DataFrame()

        student_id_column = self.find_column(
            [
                "unique id",
                "unique_id",
                "student id",
                "student_id",
                "id"
            ]
        )

        if student_id_column is None:

            print(
                "[WARNING] Student ID column not found."
            )

            return pd.DataFrame()

        mask = (
            self.dataframe[
                student_id_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
            == student_id.upper()
        )

        return self.dataframe[mask]


    # =====================================================
    # SEARCH BY NAME / FATHER NAME / ANY FIELD
    # =====================================================

    def structured_search(self, query):

        if self.dataframe is None:
            return pd.DataFrame()

        df = self.dataframe

        query_lower = query.lower()

        # -------------------------------------------------
        # Detect useful words
        # -------------------------------------------------

        search_terms = []

        # Remove common question words
        ignored_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "are",
            "of",
            "for",
            "student",
            "students",
            "details",
            "detail",
            "find",
            "show",
            "give",
            "me",
            "tell",
            "about",
            "whose",
            "has",
            "have",
            "with",
            "from",
            "and",
            "or",
            "please",
            "all",
            "information",
            "information?"
        }

        words = re.findall(
            r"[a-zA-Z0-9@._+-]+",
            query_lower
        )

        for word in words:

            if (
                word not in ignored_words
                and len(word) >= 2
            ):

                search_terms.append(word)


        # -------------------------------------------------
        # First: field-aware search
        # -------------------------------------------------

        matched_indices = set()

        for index, row in df.iterrows():

            row_text = " ".join(
                self.normalize(value)
                for value in row.values
            )

            score = 0

            for term in search_terms:

                if term in row_text:

                    score += 1

            if score > 0:

                matched_indices.add(index)


        # -------------------------------------------------
        # Return matched rows
        # -------------------------------------------------

        if matched_indices:

            results = df.loc[
                sorted(matched_indices)
            ]

            return results

        return pd.DataFrame()


    # =====================================================
    # CONVERT DATAFRAME TO TEXT
    # =====================================================

    def dataframe_to_context(
        self,
        dataframe,
        max_rows=50
    ):

        if dataframe.empty:

            return ""

        dataframe = dataframe.head(max_rows)

        records = []

        for _, row in dataframe.iterrows():

            record = []

            for column in dataframe.columns:

                value = row[column]

                record.append(
                    f"{column}: {value}"
                )

            records.append(
                "\n".join(record)
            )

        return "\n\n--------------------\n\n".join(
            records
        )


    # =====================================================
    # MAIN SEARCH
    # =====================================================

    def search_and_summarize(
        self,
        query,
        top_k=5
    ):

        print(
            f"\n[INFO] Processing query: '{query}'"
        )


        # =================================================
        # STEP 1
        # Exact Student ID
        # =================================================

        student_id = self.extract_student_id(query)

        if student_id:

            print(
                f"[INFO] Detected Student ID: {student_id}"
            )

            results = self.search_by_student_id(
                student_id
            )

            if not results.empty:

                print(
                    f"[INFO] Exact student ID match found."
                )

                context = self.dataframe_to_context(
                    results
                )

                return self.ask_gemini(
                    query,
                    context
                )

            print(
                f"[INFO] Student ID {student_id} not found."
            )

            return (
                f"No student record was found for "
                f"Unique ID {student_id}."
            )


        # =================================================
        # STEP 2
        # Structured search
        # =================================================

        structured_results = self.structured_search(
            query
        )

        if not structured_results.empty:

            print(
                f"[INFO] Structured search found "
                f"{len(structured_results)} matching records."
            )

            context = self.dataframe_to_context(
                structured_results
            )

            return self.ask_gemini(
                query,
                context
            )


        # =================================================
        # STEP 3
        # FAISS semantic search
        # =================================================

        print(
            "[INFO] No structured match found."
        )

        print(
            "[INFO] Using FAISS semantic search..."
        )

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        texts = []

        for result in results:

            metadata = result.get(
                "metadata"
            )

            if metadata:

                text = metadata.get(
                    "text",
                    ""
                )

                if text:

                    texts.append(text)


        context = "\n\n".join(texts)


        if not context:

            return "No relevant documents found."


        # =================================================
        # STEP 4
        # Gemini
        # =================================================

        return self.ask_gemini(
            query,
            context
        )


    # =====================================================
    # GEMINI
    # =====================================================

    def ask_gemini(
        self,
        query,
        context
    ):

        prompt = f"""
You are a student database assistant.

User question:
{query}

Database information:
{context}

Instructions:

1. Answer the user's question using ONLY the database information.
2. Do not invent information.
3. If multiple students match, list all relevant students.
4. If the user asks for full details, provide all available fields.
5. If the requested information is not present, clearly say so.
6. Keep the answer clear and organized.

Answer:
"""

        response = self.llm.invoke(
            [prompt]
        )

        return response.content


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    rag_search = RAGSearch()

    query = (
        "What are the full details of Zain Hassan?"
    )

    result = rag_search.search_and_summarize(
        query,
        top_k=5
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(result)