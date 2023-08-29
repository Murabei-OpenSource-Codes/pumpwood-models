rawdata_summary_query = """
    SELECT modeling_unit_id, geoarea_id, attribute_id, max(time) as max_time
    FROM database_variable
    WHERE attribute_id IN ({attributes})
    GROUP BY modeling_unit_id, geoarea_id, attribute_id
"""
