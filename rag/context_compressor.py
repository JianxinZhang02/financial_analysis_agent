from __future__ import annotations

import re

from ingestion.schema import Chunk
from rag.bm25_store import tokenize
from rag.citation import EvidenceCard


NOISE_PATTERNS = ["免责声明", "重要声明", "目录", "本报告仅供"]
CN_YEAR_DIGITS = str.maketrans({"零": "0", "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"})


class ContextCompressor:
    def __init__(self, max_evidence_chars: int = 760):
        self.max_evidence_chars = max_evidence_chars

    def compress(self, query: str, candidates: list[dict]) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        query_terms = set(tokenize(query))
        for candidate in candidates:
            chunk: Chunk = candidate["chunk"]
            if any(pattern in chunk.text[:80] for pattern in NOISE_PATTERNS):
                continue
            evidence = self._select_relevant_sentences(query, query_terms, chunk.text)
            if not evidence:
                continue
            cards.append(
                EvidenceCard(
                    claim=self._claim_from_evidence(evidence),
                    evidence=evidence,
                    source_file=chunk.metadata.get("file_name", chunk.source_file),
                    page_number=chunk.page_start,
                    chunk_id=chunk.chunk_id,
                    score=float(candidate.get("final_score", 0.0)),
                    confidence=min(0.95, 0.5 + float(candidate.get("final_score", 0.0)) / 2),
                    metadata={
                        "section_path": chunk.section_path,
                        "doc_type": chunk.doc_type,
                        "doc_id": chunk.doc_id,
                        "block_type": chunk.metadata.get("block_type", ""),
                    },
                )
            )
        return cards

    def _select_relevant_sentences(self, query: str, query_terms: set[str], text: str) -> str:
        table_context = self._select_financial_table_context(query, text)
        if table_context:
            return table_context[: self.max_evidence_chars]

        sentences = [s.strip() for s in re.split(r"(?<=[。！？；;])\s*|\n+", text) if s.strip()]
        metadata_prefixes = (
            "#",
            "文档类型",
            "研究机构",
            "发布日期",
            "公司：",
            "股票代码",
            "会议日期",
            "报告期",
        )
        sentences = [
            sentence
            for sentence in sentences
            if not any(sentence.startswith(prefix) for prefix in metadata_prefixes)
        ]
        if not sentences:
            return ""
        scored = []
        for sentence in sentences:
            terms = set(tokenize(sentence))
            overlap = len(query_terms & terms)
            has_number = bool(re.search(r"\d", sentence))
            has_financial_metric = bool(
                re.search(r"营收|营业收入|收入|净利润|毛利率|毛利|现金流|自由现金流|净现比|应收账款|市盈率|估值|风险", sentence)
            )
            score = overlap + (0.5 if has_number else 0.0) + (1.0 if has_financial_metric else 0.0)
            scored.append((sentence, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = [sentence for sentence, score in scored[:3] if score > 0]
        if not selected:
            selected = [sentences[0]]
        evidence = " ".join(selected)
        return evidence[: self.max_evidence_chars]

    def _select_financial_table_context(self, query: str, text: str) -> str:
        metric_terms = self._metric_terms(query)
        if not metric_terms:
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        for idx, line in enumerate(lines):
            if not self._is_metric_line(line, metric_terms):
                continue
            start = max(0, idx - 5)
            end = min(len(lines), idx + 4)
            context_lines = lines[start:end]
            explicit = self._explicit_table_value(query, context_lines, line)
            if explicit:
                context_lines = [explicit, *context_lines]
            return " ".join(context_lines)
        return ""

    def _metric_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        if "收入" in query or "营收" in query:
            terms.extend(["收入", "总收入", "营业收入"])
        if "毛利率" in query:
            terms.extend(["毛利率", "毛利及毛利率"])
        elif "毛利" in query:
            terms.append("毛利")
        if "利润" in query or "盈利" in query or "溢利" in query:
            terms.extend(["经营盈利", "年度盈利", "净利润", "溢利"])
        if "现金流" in query:
            terms.extend(["现金流量", "经营活动所得现金", "经营现金流"])
        return list(dict.fromkeys(terms))

    def _is_metric_line(self, line: str, metric_terms: list[str]) -> bool:
        if not re.search(r"\d", line):
            return False
        compact = re.sub(r"\s+", "", line)
        for term in metric_terms:
            if term == "收入":
                if re.match(r"^(收入|总收入|营业收入)[:：]?", compact):
                    return True
                continue
            if compact.startswith(term):
                return True
        return False

    def _explicit_table_value(self, query: str, context_lines: list[str], metric_line: str) -> str:
        target_year = self._target_year(query)
        if not target_year:
            return ""

        year_line = self._find_year_line(context_lines)
        if not year_line:
            return ""

        years = self._extract_years(year_line)
        values = self._extract_numeric_values(metric_line)
        if target_year not in years or len(values) < len(years):
            return ""

        value = values[years.index(target_year)]
        unit = self._find_unit(context_lines)
        metric = self._metric_label(metric_line)
        unit_text = f"，单位：{unit}" if unit else ""
        return f"表格解读：{target_year}年{metric}对应数值为 {value}{unit_text}。"

    def _target_year(self, query: str) -> str:
        match = re.search(r"(20\d{2})\s*年?", query)
        return match.group(1) if match else ""

    def _find_year_line(self, lines: list[str]) -> str:
        for line in lines:
            if len(self._extract_years(line)) >= 2:
                return line
        return ""

    def _extract_years(self, line: str) -> list[str]:
        years = re.findall(r"(20\d{2})\s*年?", line)
        cn_years = re.findall(r"二[零〇][零〇一二三四五六七八九]{2}年", line)
        for item in cn_years:
            digits = item[:-1].translate(CN_YEAR_DIGITS)
            if re.fullmatch(r"20\d{2}", digits):
                years.append(digits)
        return years

    def _find_unit(self, lines: list[str]) -> str:
        for line in lines:
            if "人民币百万元" in line:
                return "人民币百万元"
            if "人民币千元" in line:
                return "人民币千元"
            if "人民币亿元" in line:
                return "人民币亿元"
        return ""

    def _extract_numeric_values(self, line: str) -> list[str]:
        values: list[str] = []
        for match in re.findall(r"-?\(?\d[\d,]*(?:\.\d+)?\)?", line):
            value = match.strip("()")
            if "," in value:
                groups = value.lstrip("-").split(",")
                if len(groups) > 1 and not all(len(group) == 3 for group in groups[1:]):
                    continue
            values.append(match)
        return values

    def _metric_label(self, metric_line: str) -> str:
        compact = re.sub(r"\s+", " ", metric_line).strip()
        return compact.split(" ")[0].strip("：:")

    def _claim_from_evidence(self, evidence: str) -> str:
        return evidence.split("。")[0].strip()[:160]
