def get_good_rate_comps(df, label):

    big_comps = df['Name'].value_counts()[:100].index
    good_rate_comps = []

    for comp in big_comps:
        val_counts = df[df['Name'] == comp][label].value_counts(normalize=True)
        try:
            chrgoff_rate = val_counts[1]
        except IndexError:
            chrgoff_rate = 0
            print(comp)
            print(val_counts)

        if chrgoff_rate < 0.10:
            good_rate_comps.append(comp)


    print(f'n of good_rate_comps: {len(good_rate_comps)}')

    return good_rate_comps
