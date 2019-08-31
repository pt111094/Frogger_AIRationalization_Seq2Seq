import xlsxwriter


if __name__ == '__main__':

    workbook = xlsxwriter.Workbook('Turk_Master_File.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.set_column('B:B', 35)
    worksheet.write('A2', 'Insert an image in a cell:')
    worksheet.insert_image('B2', 'Screenshot_0.png', {'x_scale': 0.5, 'y_scale': 0.5})
    workbook.close()
