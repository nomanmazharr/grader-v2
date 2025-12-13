import streamlit as st
import os
import tempfile
from main import grade_and_annotate_student, extract_question_text, extract_marking_rubric
import traceback

def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temp directory."""
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def parse_pages(page_str):
    """Parse comma-separated page numbers."""
    if not page_str:
        return []
    return [int(p.strip()) for p in page_str.split(',') if p.strip().isdigit()]

def main():
    st.set_page_config(page_title="Exam Grader", page_icon="📚", layout="wide")
    st.title("📚 Automated Exam Grader")
    
    # Temp directory for uploaded files
    temp_dir = tempfile.mkdtemp(prefix="exam_grader_")
    
    # File uploads
    st.header("📁 Upload PDFs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        question_pdf = st.file_uploader("📋 Question Paper", type="pdf")
    with col2:
        model_answer_pdf = st.file_uploader("✅ Model Answers", type="pdf")
    with col3:
        student_pdf = st.file_uploader("👤 Student Assignment", type="pdf")
    
    # Configuration
    st.header("⚙️ Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        question_pages_str = st.text_input("Question Pages", value="3")
        answer_pages_str = st.text_input("Model Answer Pages", value="3,4,5")
        question_num = st.text_input("Question Number", value="1")
        student_pages_str = st.text_input("Student Pages", value="1,2,3")
        
    with col2:
        student_name = st.text_input("Student Name", value="Student")
        output_dir = st.text_input("Output Directory", value="annotations")
    
    # Parse pages
    question_pages = parse_pages(question_pages_str)
    answer_pages = parse_pages(answer_pages_str)
    student_pages = parse_pages(student_pages_str)
    
    if st.checkbox("🐛 Show Debug Info"):
        st.write("**Uploaded Files:**")
        if question_pdf:
            st.write(f"- Question: {question_pdf.name} ({len(question_pdf.getbuffer()):,} bytes)")
        if model_answer_pdf:
            st.write(f"- Model: {model_answer_pdf.name} ({len(model_answer_pdf.getbuffer()):,} bytes)")
        if student_pdf:
            st.write(f"- Student: {student_pdf.name} ({len(student_pdf.getbuffer()):,} bytes)")
        
        st.write("**Parsed Pages:**")
        st.write(f"- Question pages: {question_pages}")
        st.write(f"- Answer pages: {answer_pages}")
        st.write(f"- Student pages: {student_pages}")
    
    # Process button
    if st.button("🚀 Process Exam", type="primary"):
        if not all([question_pdf, model_answer_pdf, student_pdf]):
            st.error("❌ Upload all three PDFs!")
            st.stop()
        
        if not all([question_pages, answer_pages, student_pages]):
            st.error("❌ Specify valid page numbers!")
            st.stop()
        
        # Progress container
        progress_container = st.container()
        status_container = st.container()
        
        try:
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Step 1: Save files
            status_text.info("💾 Saving uploaded files...")
            progress_bar.progress(10)
            
            question_path = save_uploaded_file(question_pdf, temp_dir)
            model_path = save_uploaded_file(model_answer_pdf, temp_dir)
            student_path = save_uploaded_file(student_pdf, temp_dir)
            
            # Verify files saved
            if not all(os.path.exists(p) for p in [question_path, model_path, student_path]):
                st.error("❌ File save failed! Check permissions.")
                st.stop()
            
            progress_bar.progress(20)
            
            # Create output dir
            os.makedirs(output_dir, exist_ok=True)
            status_text.success(f"📁 Output directory: {output_dir}")
            
            # Step 2: Extract (manual call for debugging)
            status_text.info("🔍 Extracting questions and model answers...")
            progress_bar.progress(30)
            
            extract_success, questions_json = extract_question_text(
                question_path, question_pages, question_num
            )
            
            if not extract_success:
                st.error("❌ Extraction failed!")
                progress_bar.progress(0)
                st.stop()
            
            status_text.success(f"✅ Extracted JSONs:")
            st.write(f"   📄 Questions: {questions_json}")
            
            if not os.path.exists(questions_json):
                st.error("❌ Questions JSON file not created!")
                st.stop()

            progress_bar.progress(30)
            

            # Step 3: Merging marking criteria
            status_text.info("🧩 Extracting and merging marking criteria...")
            
            answer_success, model_path = extract_marking_rubric(
                model_path, answer_pages
            )

            if not answer_success:
                st.error("Failed to extract marking criteria.")
            else:
                st.success("Marking criteria successfully extracted.")
                # model_json = merged_model_json  # Use
            progress_bar.progress(60)

            # Step 3: Grade and annotate
            status_text.info("Grading and annotating student assignment...")
            
            
            grade_success, grade_message, annotated_path = grade_and_annotate_student(
                student_path, student_name, questions_json, model_path, 
                question_num, student_pages, output_dir
            )
            progress_bar.progress(100)
            
            if grade_success:
                status_text.success("🎉 Processing completed!")
                
                # Check if annotated PDF exists
                if os.path.exists(annotated_path):
                    st.success(f"✅ Annotated PDF saved: {annotated_path}")
                    
                    # Download button
                    with open(annotated_path, "rb") as f:
                        st.download_button(
                            "📥 Download Annotated PDF",
                            f.read(),
                            file_name=f"{student_name}_annotated.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("⚠️ Processing succeeded but annotated PDF not found!")
                    st.info(f"Expected path: {annotated_path}")
                    
            else:
                st.error(f"❌ Grading failed: {grade_message}")
                
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
            
        finally:
            progress_bar.empty()
            status_text.empty()

    # File info
    if question_pdf:
        st.info(f"📋 {question_pdf.name} ({question_pdf.size/1024:.1f}KB)")
    if model_answer_pdf:
        st.info(f"✅ {model_answer_pdf.name} ({model_answer_pdf.size/1024:.1f}KB)")
    if student_pdf:
        st.info(f"👤 {student_pdf.name} ({student_pdf.size/1024:.1f}KB)")

if __name__ == "__main__":
    main()