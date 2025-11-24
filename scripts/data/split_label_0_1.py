#!/usr/bin/env python3
"""
Script để tách các sample theo label 0 và label 1
Tạo 2 file:
- File chứa label = 0
- File chứa label = 1
Load theo chunk để tránh tràn RAM
"""

import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import sys
import gc

def split_parquet_by_label_0_1(input_path, label_col='Label', batch_size=10000):
    """
    Tách file parquet thành 2 file dựa trên label 0 và 1
    
    Args:
        input_path: Đường dẫn đến file parquet gốc
        label_col: Tên cột label
        batch_size: Kích thước batch khi xử lý
    """
    input_path = Path(input_path)
    
    # Tạo tên file output
    base_name = input_path.stem
    output_dir = input_path.parent
    
    output_label0 = output_dir / f"{base_name}_label_0.parquet"
    output_label1 = output_dir / f"{base_name}_label_1.parquet"
    
    print(f"📂 Đang xử lý file: {input_path}")
    print(f"   Output file label 0: {output_label0}")
    print(f"   Output file label 1: {output_label1}")
    
    # Mở file parquet
    pq_file = pq.ParquetFile(input_path)
    total_rows = pq_file.metadata.num_rows
    print(f"   Tổng số dòng: {total_rows:,}")
    
    # Lấy tất cả các cột và schema
    all_columns = pq_file.schema.names
    file_schema = pq_file.schema_arrow
    
    # Kiểm tra xem có cột label không
    if label_col not in all_columns:
        # Thử tìm cột label với tên khác
        label_candidates = [col for col in all_columns if 'label' in col.lower()]
        if label_candidates:
            label_col = label_candidates[0]
            print(f"   ⚠️  Tìm thấy cột label: {label_col}")
        else:
            raise ValueError(f"Không tìm thấy cột label trong file. Các cột có sẵn: {all_columns[:10]}")
    
    # Đếm số lượng
    count_label0 = 0
    count_label1 = 0
    
    # Writers để ghi trực tiếp (streaming) thay vì lưu tất cả vào RAM
    writer_label0 = None
    writer_label1 = None
    
    # Xử lý theo batch
    print(f"   🔄 Đang đọc và tách dữ liệu theo chunk...")
    batch_num = 0
    
    for batch in pq_file.iter_batches(batch_size=batch_size, columns=all_columns):
        batch_num += 1
        if batch_num % 100 == 0:
            processed = min(batch_num * batch_size, total_rows)
            progress = min(100, (processed / total_rows) * 100)
            print(f"   ⏳ Đã xử lý: {processed:,}/{total_rows:,} dòng ({progress:.1f}%)")
        
        # Chuyển batch sang pandas DataFrame
        df_batch = batch.to_pandas()
        
        # Tách dựa trên label
        mask_label0 = df_batch[label_col] == 0
        mask_label1 = df_batch[label_col] == 1
        
        df_label0 = df_batch[mask_label0]
        df_label1 = df_batch[mask_label1]
        
        # Đếm
        count_label0 += len(df_label0)
        count_label1 += len(df_label1)
        
        # Ghi trực tiếp vào file (streaming) để tránh tràn RAM
        if len(df_label0) > 0:
            table_label0 = pa.Table.from_pandas(df_label0, preserve_index=False)
            if writer_label0 is None:
                # Khởi tạo writer lần đầu
                writer_label0 = pq.ParquetWriter(output_label0, table_label0.schema)
            writer_label0.write_table(table_label0)
            del table_label0
        
        if len(df_label1) > 0:
            table_label1 = pa.Table.from_pandas(df_label1, preserve_index=False)
            if writer_label1 is None:
                # Khởi tạo writer lần đầu
                writer_label1 = pq.ParquetWriter(output_label1, table_label1.schema)
            writer_label1.write_table(table_label1)
            del table_label1
        
        # Giải phóng memory
        del df_batch, df_label0, df_label1
        if batch_num % 100 == 0:
            gc.collect()
    
    # Đóng writers
    if writer_label0 is not None:
        writer_label0.close()
    if writer_label1 is not None:
        writer_label1.close()
    
    print(f"   ✅ Đã xử lý xong!")
    print(f"   📊 Thống kê:")
    print(f"      - Label = 0: {count_label0:,} dòng")
    print(f"      - Label = 1: {count_label1:,} dòng")
    print(f"      - Tổng: {count_label0 + count_label1:,} dòng")
    
    # Kiểm tra và tạo file rỗng nếu cần
    if count_label0 == 0:
        print(f"   ⚠️  Không có dữ liệu với label = 0, tạo file rỗng: {output_label0.name}")
        empty_arrays = [pa.array([], type=field.type) for field in file_schema]
        empty_table = pa.Table.from_arrays(empty_arrays, schema=file_schema)
        pq.write_table(empty_table, output_label0)
        print(f"   ✅ Đã tạo file rỗng: {output_label0.name}")
    else:
        print(f"   ✅ Đã ghi {count_label0:,} dòng vào {output_label0.name}")
    
    if count_label1 == 0:
        print(f"   ⚠️  Không có dữ liệu với label = 1, tạo file rỗng: {output_label1.name}")
        empty_arrays = [pa.array([], type=field.type) for field in file_schema]
        empty_table = pa.Table.from_arrays(empty_arrays, schema=file_schema)
        pq.write_table(empty_table, output_label1)
        print(f"   ✅ Đã tạo file rỗng: {output_label1.name}")
    else:
        print(f"   ✅ Đã ghi {count_label1:,} dòng vào {output_label1.name}")
    
    print(f"   ✅ Hoàn thành xử lý file: {input_path.name}\n")
    
    return output_label0, output_label1, count_label0, count_label1

def main():
    # File cần xử lý
    input_file = "/home/hytong/Documents/model_extraction_malware/data/ember_2018_v2/train/train_ember_2018_v2_features_label_other.parquet"
    
    print("=" * 70)
    print("🔀 TÁCH FILE PARQUET THEO LABEL 0 VÀ LABEL 1")
    print("=" * 70)
    print()
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File không tồn tại: {input_path}")
        sys.exit(1)
    
    try:
        output_label0, output_label1, count_label0, count_label1 = split_parquet_by_label_0_1(
            input_path, 
            label_col='Label',
            batch_size=10000  # Điều chỉnh batch_size nếu cần (nhỏ hơn nếu RAM ít)
        )
        
        # Tóm tắt kết quả
        print("=" * 70)
        print("📊 TÓM TẮT KẾT QUẢ")
        print("=" * 70)
        print(f"\n📁 File gốc: {input_path.name}")
        print(f"   - {output_label0.name}: {count_label0:,} dòng")
        print(f"   - {output_label1.name}: {count_label1:,} dòng")
        print("=" * 70)
        print("✅ Hoàn thành!")
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

