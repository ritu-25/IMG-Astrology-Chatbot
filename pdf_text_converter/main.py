# from pipeline import BPHSPipeline
# from config import Config


# def main():
#     config = Config()

#     pdf_path = "sample.pdf"  # 🔴 CHANGE THIS
#     start_page = None
#     end_page = None

#     print("\n" + "="*70)
#     print("🚀 PDF TEXT CONVERTER WITH TABLE DETECTION")
#     print("="*70)

#     print("\nSelect Review Mode:")
#     print("1 → GUI Review (recommended)")
#     print("2 → Quick Review (CLI)")
#     print("3 → Auto (no review)")

#     while True:
#         choice = input("\nEnter choice [1/2/3]: ").strip()
#         if choice == "1":
#             review_mode = "gui"
#             break
#         elif choice == "2":
#             review_mode = "quick"
#             break
#         elif choice == "3":
#             review_mode = "auto"
#             break
#         else:
#             print("❌ Invalid choice")

#     try:
#         pipeline = BPHSPipeline(config=config, review_mode=review_mode)

#         pipeline.process_pdf(
#             pdf_path=pdf_path,
#             start_page=start_page,
#             end_page=end_page
#         )

#         print("\n✅ PROCESSING COMPLETE")
#         print(f"📁 Output directory: {config.OUTPUT_DIR}")
        
#     except Exception as e:
#         print(f"\n❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()



"""
Main script to run the pipeline with GUI-based manual review
"""

from pipeline import BPHSPipeline
from config import Config

def main():
    # Initialize config
    config = Config()
    
    # Configure your PDF
    pdf_path = "sample.pdf"  # CHANGE THIS TO YOUR PDF
    start_page = None  # Optional: start page (None = from beginning)
    end_page = None    # Optional: end page (None = to end)
    
    print("\n" + "="*70)
    print("🚀 BPHS PIPELINE WITH GUI MANUAL REVIEW")
    print("="*70)
    print("\n📋 REVIEW MODE OPTIONS:")
    print("  1. GUI Mode (Recommended) - Interactive visual review")
    print("     • Click on tables to select them")
    print("     • Press M to mask, K to keep")
    print("     • Draw your own boxes around missed tables")
    print("     • ⭐ NEW: ALL pages shown (even with no auto-detected tables)")
    print()
    print("  2. Quick Mode - Command line with preview images")
    print("     • Review preview images saved to disk")
    print("     • Make decisions via keyboard input")
    print("     • Good for remote/SSH sessions")
    print()
    print("  3. Auto Mode - No review (mask all detected tables)")
    print("     • Fastest option")
    print("     • Use when you trust automatic detection")
    print()
    
    while True:
        choice = input("Select mode [1/2/3]: ").strip()
        
        if choice == '1':
            review_mode = 'gui'
            print("\n✅ GUI review mode selected")
            print("   → You'll see an interactive window with table overlays")
            break
        elif choice == '2':
            review_mode = 'quick'
            print("\n✅ Quick review mode selected")
            print("   → Preview images will be saved, review via command line")
            break
        elif choice == '3':
            review_mode = 'auto'
            print("\n✅ Auto mode selected (no manual review)")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3")
    
    print("\n" + "="*70)
    print("INITIALIZING PIPELINE...")
    print("="*70)
    
    # Initialize pipeline with chosen mode
    pipeline = BPHSPipeline(
        config=config,
        review_mode=review_mode
    )
    
    # Process PDF
    try:
        print(f"\n📄 Processing: {pdf_path}")
        if start_page or end_page:
            print(f"   Pages: {start_page or 'start'} to {end_page or 'end'}")
        
        results = pipeline.process_pdf(
            pdf_path=pdf_path,
            start_page=start_page,
            end_page=end_page
        )
        
        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Output directory: {config.OUTPUT_DIR}/")
        print("\n📄 Generated files:")
        print("   • extracted_text.txt - Page text with approved tables masked")
        print("   • masked_table_text.txt - Text from masked tables only")
        print("   • extraction_results.json - Detailed results with metadata")
        
        if review_mode == 'gui':
            print("   • review_gui/gui_review_log.json - Your review decisions")
            print("   • page_XXX_masked_viz.png - Visualization of masked tables")
        elif review_mode == 'quick':
            print("   • review_quick/quick_review_log.json - Your review decisions")
            print("   • review_quick/page_XXX_preview.png - Preview images")
        
        print("\n💡 Tip: Check masked_table_text.txt to see what was removed from pages")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        print("   Partial results may have been saved to output directory")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"   Make sure '{pdf_path}' exists")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck the error above and try again")


if __name__ == "__main__":
    main()