#!/usr/bin/env python3
"""
Script để tách các sample có label = -1 ra file riêng
Tạo 2 file cho mỗi file gốc:
- File chứa label = -1
- File chứa label != -1
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import sys

def split_parquet_by_label(input_path, label_col='Label', batch_size=10000):
    """
    Tách file parquet thành 2 file dựa trên label = -1
    
    Args:
        input_path: Đường dẫn đến file parquet gốc
        label_col: Tên cột label
        batch_size: Kích thước batch khi xử lý
    """
    input_path = Path(input_path)
    
    # Tạo tên file output
    base_name = input_path.stem
    output_dir = input_path.parent
    
    output_minus1 = output_dir / f"{base_name}_label_minus1.parquet"
    output_other = output_dir / f"{base_name}_label_other.parquet"
    
    print(f"📂 Đang xử lý file: {input_path}")
    print(f"   Output file 1: {output_minus1}")
    print(f"   Output file 2: {output_other}")
    
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
    count_minus1 = 0
    count_other = 0
    
    # Danh sách để lưu các batch
    batches_minus1 = []
    batches_other = []
    
    # Xử lý theo batch
    print(f"   🔄 Đang đọc và tách dữ liệu...")
    batch_num = 0
    
    for batch in pq_file.iter_batches(batch_size=batch_size, columns=all_columns):
        batch_num += 1
        if batch_num % 100 == 0:
            processed = (batch_num * batch_size)
            progress = min(100, (processed / total_rows) * 100)
            print(f"   ⏳ Đã xử lý: {processed:,}/{total_rows:,} dòng ({progress:.1f}%)")
        
        # Chuyển batch sang pandas DataFrame
        df_batch = batch.to_pandas()
        
        # Tách dựa trên label
        mask_minus1 = df_batch[label_col] == -1
        df_minus1 = df_batch[mask_minus1]
        df_other = df_batch[~mask_minus1]
        
        # Đếm
        count_minus1 += len(df_minus1)
        count_other += len(df_other)
        
        # Lưu vào danh sách batch
        if len(df_minus1) > 0:
            batches_minus1.append(pa.Table.from_pandas(df_minus1, preserve_index=False))
        if len(df_other) > 0:
            batches_other.append(pa.Table.from_pandas(df_other, preserve_index=False))
    
    print(f"   ✅ Đã xử lý xong!")
    print(f"   📊 Thống kê:")
    print(f"      - Label = -1: {count_minus1:,} dòng")
    print(f"      - Label != -1: {count_other:,} dòng")
    print(f"      - Tổng: {count_minus1 + count_other:,} dòng")
    
    # Ghi file chứa label = -1 (luôn tạo file, kể cả khi rỗng)
    if count_minus1 > 0 and batches_minus1:
        print(f"   💾 Đang ghi file: {output_minus1.name}")
        table_minus1 = pa.concat_tables(batches_minus1)
        pq.write_table(table_minus1, output_minus1)
        print(f"   ✅ Đã ghi {count_minus1:,} dòng vào {output_minus1.name}")
        del batches_minus1
        del table_minus1
    else:
        # Tạo file rỗng với schema đúng
        print(f"   ⚠️  Không có dữ liệu với label = -1, tạo file rỗng: {output_minus1.name}")
        # Tạo bảng rỗng với schema của file gốc
        empty_arrays = [pa.array([], type=field.type) for field in file_schema]
        empty_table = pa.Table.from_arrays(empty_arrays, schema=file_schema)
        pq.write_table(empty_table, output_minus1)
        print(f"   ✅ Đã tạo file rỗng: {output_minus1.name}")
        if batches_minus1:
            del batches_minus1
    
    # Ghi file chứa label != -1 (luôn tạo file, kể cả khi rỗng)
    if count_other > 0 and batches_other:
        print(f"   💾 Đang ghi file: {output_other.name}")
        table_other = pa.concat_tables(batches_other)
        pq.write_table(table_other, output_other)
        print(f"   ✅ Đã ghi {count_other:,} dòng vào {output_other.name}")
        del batches_other
        del table_other
    else:
        # Tạo file rỗng với schema đúng
        print(f"   ⚠️  Không có dữ liệu với label != -1, tạo file rỗng: {output_other.name}")
        # Tạo bảng rỗng với schema của file gốc
        empty_arrays = [pa.array([], type=field.type) for field in file_schema]
        empty_table = pa.Table.from_arrays(empty_arrays, schema=file_schema)
        pq.write_table(empty_table, output_other)
        print(f"   ✅ Đã tạo file rỗng: {output_other.name}")
        if batches_other:
            del batches_other
    
    print(f"   ✅ Hoàn thành xử lý file: {input_path.name}\n")
    
    return output_minus1, output_other, count_minus1, count_other

def main():
    # Danh sách file cần xử lý
    files_to_process = [
        "/home/hytong/Documents/model_extraction_malware/src/test_ember_2018_v2_features.parquet",
        "/home/hytong/Documents/model_extraction_malware/src/train_ember_2018_v2_features.parquet"
    ]
    
    print("=" * 70)
    print("🔀 TÁCH FILE PARQUET THEO LABEL = -1")
    print("=" * 70)
    print()
    
    results = []
    
    for file_path in files_to_process:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File không tồn tại: {file_path}")
            continue
        
        try:
            output_minus1, output_other, count_minus1, count_other = split_parquet_by_label(
                file_path, 
                label_col='Label',
                batch_size=10000
            )
            results.append({
                'file': file_path.name,
                'minus1_file': output_minus1.name,
                'other_file': output_other.name,
                'minus1_count': count_minus1,
                'other_count': count_other
            })
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Tóm tắt kết quả
    print("=" * 70)
    print("📊 TÓM TẮT KẾT QUẢ")
    print("=" * 70)
    for result in results:
        print(f"\n📁 File: {result['file']}")
        print(f"   - {result['minus1_file']}: {result['minus1_count']:,} dòng")
        print(f"   - {result['other_file']}: {result['other_count']:,} dòng")
    print("=" * 70)
    print("✅ Hoàn thành tất cả!")

if __name__ == "__main__":
    main()

