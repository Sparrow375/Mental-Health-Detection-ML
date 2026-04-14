import os

main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'

with open(main_file, 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = '        // ── System 1'
end_marker = 'item { Spacer(Modifier.height(16.dp)) }\n'

start_idx = text.find(start_marker)
end_idx = text.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_marker)
    items_str = text[start_idx:end_idx]
    
    # remove from current location
    text = text[:start_idx] + text[end_idx:]
    
    # Find insertion point: at the end of the LazyColumn inside HomeScreen
    sys_marker = 'MetricPill("📶 Mobile Data"'
    sys_idx = text.find(sys_marker)
    if sys_idx != -1:
        # find the end of this item block
        # The structure is:
        # item {
        #     InfoCard(...) {
        #         Row(...) {
        #             MetricPill(...)
        #             MetricPill(...)
        #         }
        #     }
        # }
        
        # We need to find the matching '}' for item { or just find the '}' that closes the item.
        # Just looking for '            }' which closes item {
        item_end_marker = '            }\n'
        item_end_idx = text.find(item_end_marker, sys_idx)
        if item_end_idx != -1:
            item_end_idx += len(item_end_marker)
            text = text[:item_end_idx] + '\n' + items_str + text[item_end_idx:]
            
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print('Fixed MainActivity.kt')
        else:
            print('Could not find item_end_marker')
    else:
        print('Could not find sys_marker')
else:
    print('Could not find markers')
